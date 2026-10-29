// microholdem C++ hot path: hand evaluation, Monte Carlo equity, ICM.
//
// Card encoding (shared with the Python package):
//   card = rank * 4 + suit, rank 0..12 (2..A), suit 0..3 (c, d, h, s).
//   A hand is a 64-bit mask with one 16-bit lane per suit and one bit per
//   rank inside the lane: bit = suit * 16 + rank.
//
// Evaluation returns a 32-bit value. Larger means stronger. The top four
// bits (26..29) hold the category; the low 26 bits order hands inside the
// category. Values are comparable across 5, 6 and 7 card hands.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

constexpr uint32_t CAT_SHIFT = 26;
constexpr uint32_t HIGH_CARD = 0, ONE_PAIR = 1, TWO_PAIR = 2, TRIPS = 3, STRAIGHT = 4,
                   FLUSH = 5, FULL_HOUSE = 6, QUADS = 7, STRAIGHT_FLUSH = 8;

// ---------------------------------------------------------------------------
// Rank-mask tables (13-bit masks, 8192 entries each).
// ---------------------------------------------------------------------------
struct RankTables {
    uint16_t top1[8192];
    uint16_t top2[8192];
    uint16_t top3[8192];
    uint16_t top5[8192];
    uint8_t straight[8192];  // 0 = none, else (top rank index + 1)

    static uint16_t keep_top(uint32_t m, int k) {
        while (__builtin_popcount(m) > k) m &= m - 1;  // clear lowest set bit
        return static_cast<uint16_t>(m);
    }
    static uint8_t find_straight(uint32_t m) {
        for (int top = 12; top >= 4; --top) {
            uint32_t need = 0x1Fu << (top - 4);
            if ((m & need) == need) return static_cast<uint8_t>(top + 1);
        }
        if ((m & 0x100Fu) == 0x100Fu) return 4;  // A-2-3-4-5, five high
        return 0;
    }
    RankTables() {
        for (uint32_t m = 0; m < 8192; ++m) {
            top1[m] = keep_top(m, 1);
            top2[m] = keep_top(m, 2);
            top3[m] = keep_top(m, 3);
            top5[m] = keep_top(m, 5);
            straight[m] = find_straight(m);
        }
    }
};
const RankTables T;

inline uint64_t card_bit(int c) { return 1ULL << ((c & 3) * 16 + (c >> 2)); }
inline int top_index(uint32_t x) { return 31 - __builtin_clz(x); }

inline uint32_t evaluate_mask(uint64_t m) {
    const uint32_t s0 = static_cast<uint32_t>(m & 0x1FFFu);
    const uint32_t s1 = static_cast<uint32_t>((m >> 16) & 0x1FFFu);
    const uint32_t s2 = static_cast<uint32_t>((m >> 32) & 0x1FFFu);
    const uint32_t s3 = static_cast<uint32_t>((m >> 48) & 0x1FFFu);

    // With at most seven cards only one suit can hold five or more cards, and a
    // flush then dominates every non-flush category that is still possible.
    uint32_t fs = 0;
    if (__builtin_popcount(s0) >= 5) fs = s0;
    else if (__builtin_popcount(s1) >= 5) fs = s1;
    else if (__builtin_popcount(s2) >= 5) fs = s2;
    else if (__builtin_popcount(s3) >= 5) fs = s3;
    if (fs) {
        const uint32_t st = T.straight[fs];
        if (st) return (STRAIGHT_FLUSH << CAT_SHIFT) | st;
        return (FLUSH << CAT_SHIFT) | T.top5[fs];
    }

    const uint32_t any = s0 | s1 | s2 | s3;
    const uint32_t pairs = (s0 & s1) | (s0 & s2) | (s0 & s3) | (s1 & s2) | (s1 & s3) | (s2 & s3);
    const uint32_t trips = (s0 & s1 & s2) | (s0 & s1 & s3) | (s0 & s2 & s3) | (s1 & s2 & s3);
    const uint32_t quads = s0 & s1 & s2 & s3;

    if (quads) {
        const int q = top_index(quads);
        const uint32_t kick = any & ~quads;
        return (QUADS << CAT_SHIFT) | (static_cast<uint32_t>(q) << 13) | T.top1[kick];
    }
    if (trips) {
        const int t = top_index(trips);
        const uint32_t rest = pairs & ~(1u << t);
        if (rest) return (FULL_HOUSE << CAT_SHIFT) | (static_cast<uint32_t>(t) << 13) | T.top1[rest];
    }
    const uint32_t st = T.straight[any];
    if (st) return (STRAIGHT << CAT_SHIFT) | st;
    if (trips) {
        const int t = top_index(trips);
        const uint32_t kick = any & ~(1u << t);
        return (TRIPS << CAT_SHIFT) | (static_cast<uint32_t>(t) << 13) | T.top2[kick];
    }
    if (pairs) {
        if (__builtin_popcount(pairs) >= 2) {
            const uint32_t two = T.top2[pairs];
            const uint32_t kick = any & ~two;
            return (TWO_PAIR << CAT_SHIFT) | (two << 13) | T.top1[kick];
        }
        const int p = top_index(pairs);
        const uint32_t kick = any & ~pairs;
        return (ONE_PAIR << CAT_SHIFT) | (static_cast<uint32_t>(p) << 13) | T.top3[kick];
    }
    return (HIGH_CARD << CAT_SHIFT) | T.top5[any];
}

// ---------------------------------------------------------------------------
// Deterministic RNG (xoshiro256**) and a reusable deck.
// ---------------------------------------------------------------------------
struct Rng {
    uint64_t s[4];
    explicit Rng(uint64_t seed) {
        uint64_t z = seed;
        for (auto& v : s) {
            z += 0x9E3779B97F4A7C15ULL;
            uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            v = x ^ (x >> 31);
        }
    }
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        const uint64_t r = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return r;
    }
    // Uniform integer in [0, n). Bias is below 2^-32 for the n used here.
    inline uint32_t below(uint32_t n) { return static_cast<uint32_t>(((next() >> 32) * static_cast<uint64_t>(n)) >> 32); }
    inline double unit() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
};

struct Deck {
    int cards[52];
    int n = 0;
    void build(uint64_t excluded) {
        n = 0;
        for (int c = 0; c < 52; ++c)
            if (!(excluded & card_bit(c))) cards[n++] = c;
    }
    // Partial Fisher-Yates: after the call, cards[0..k) is a uniform random
    // k-subset in random order. The permuted state is kept between calls.
    inline void draw(int k, Rng& rng) {
        for (int i = 0; i < k; ++i) {
            const int j = i + static_cast<int>(rng.below(static_cast<uint32_t>(n - i)));
            std::swap(cards[i], cards[j]);
        }
    }
};

uint64_t mask_of(const std::vector<int>& cards, const char* what) {
    uint64_t m = 0;
    for (int c : cards) {
        if (c < 0 || c > 51) throw std::invalid_argument(std::string(what) + ": card out of range");
        const uint64_t b = card_bit(c);
        if (m & b) throw std::invalid_argument(std::string(what) + ": duplicate card");
        m |= b;
    }
    return m;
}

// ---------------------------------------------------------------------------
// Monte Carlo equity.
// ---------------------------------------------------------------------------
double equity_vs_random(const std::vector<int>& hero, const std::vector<int>& board, int n_opp,
                        int n_samples, uint64_t seed) {
    if (hero.size() != 2) throw std::invalid_argument("hero must hold exactly two cards");
    if (board.size() > 5) throw std::invalid_argument("board holds at most five cards");
    if (n_opp < 1 || n_opp > 9) throw std::invalid_argument("n_opp must be in [1, 9]");
    if (n_samples < 1) throw std::invalid_argument("n_samples must be positive");
    const uint64_t hm = mask_of(hero, "hero");
    const uint64_t bm = mask_of(board, "board");
    if (hm & bm) throw std::invalid_argument("hero and board share a card");
    Deck deck;
    deck.build(hm | bm);
    Rng rng(seed);
    const int need = 5 - static_cast<int>(board.size());
    const int k = need + 2 * n_opp;
    double total = 0.0;
    for (int s = 0; s < n_samples; ++s) {
        deck.draw(k, rng);
        uint64_t b = bm;
        for (int i = 0; i < need; ++i) b |= card_bit(deck.cards[i]);
        const uint32_t hv = evaluate_mask(hm | b);
        int tied = 0;
        bool beaten = false;
        for (int o = 0; o < n_opp; ++o) {
            const uint64_t om = card_bit(deck.cards[need + 2 * o]) | card_bit(deck.cards[need + 2 * o + 1]);
            const uint32_t ov = evaluate_mask(om | b);
            if (ov > hv) { beaten = true; break; }
            if (ov == hv) ++tied;
        }
        if (!beaten) total += 1.0 / (1.0 + tied);
    }
    return total / n_samples;
}

struct WeightedRange {
    std::vector<int> c1, c2;
    std::vector<double> cum;  // cumulative weights, last == total
    double total = 0.0;
    inline int sample(Rng& rng) const {
        const double u = rng.unit() * total;
        return static_cast<int>(std::lower_bound(cum.begin(), cum.end(), u) - cum.begin());
    }
};

WeightedRange make_range(const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& combos,
                         const py::array_t<double, py::array::c_style | py::array::forcecast>& weights) {
    if (combos.ndim() != 2 || combos.shape(1) != 2) throw std::invalid_argument("combos must have shape (M, 2)");
    if (weights.ndim() != 1 || weights.shape(0) != combos.shape(0)) throw std::invalid_argument("weights must have shape (M,)");
    WeightedRange r;
    const auto cb = combos.unchecked<2>();
    const auto wb = weights.unchecked<1>();
    double acc = 0.0;
    for (py::ssize_t i = 0; i < combos.shape(0); ++i) {
        const int a = cb(i, 0), b = cb(i, 1);
        if (a < 0 || a > 51 || b < 0 || b > 51 || a == b) throw std::invalid_argument("invalid combo in range");
        const double w = wb(i);
        if (w < 0.0) throw std::invalid_argument("negative weight in range");
        acc += w;
        r.c1.push_back(a);
        r.c2.push_back(b);
        r.cum.push_back(acc);
    }
    if (acc <= 0.0) throw std::invalid_argument("range has zero total weight");
    r.total = acc;
    return r;
}

// Hero equity against one or more opponents who each hold a weighted range.
// Combos that collide with known cards or other sampled opponents are
// rejected and redrawn, so the sample is uniform over consistent deals.
double equity_vs_ranges(const std::vector<int>& hero, const std::vector<int>& board,
                        const std::vector<WeightedRange>& ranges, int n_samples, uint64_t seed) {
    if (hero.size() != 2) throw std::invalid_argument("hero must hold exactly two cards");
    if (board.size() > 5) throw std::invalid_argument("board holds at most five cards");
    if (ranges.empty() || ranges.size() > 9) throw std::invalid_argument("need between 1 and 9 opponent ranges");
    if (n_samples < 1) throw std::invalid_argument("n_samples must be positive");
    const uint64_t hm = mask_of(hero, "hero");
    const uint64_t bm = mask_of(board, "board");
    if (hm & bm) throw std::invalid_argument("hero and board share a card");
    const uint64_t known = hm | bm;
    const int n_opp = static_cast<int>(ranges.size());
    Rng rng(seed);
    const int need = 5 - static_cast<int>(board.size());
    Deck deck;
    std::array<uint64_t, 9> om{};
    double total = 0.0;
    for (int s = 0; s < n_samples; ++s) {
        uint64_t used = known;
        for (int o = 0; o < n_opp; ++o) {
            const WeightedRange& r = ranges[o];
            int tries = 0;
            while (true) {
                const int idx = r.sample(rng);
                const uint64_t m = card_bit(r.c1[idx]) | card_bit(r.c2[idx]);
                if (!(m & used)) { om[o] = m; used |= m; break; }
                if (++tries > 10000) throw std::runtime_error("range is blocked by known cards");
            }
        }
        deck.build(used);
        deck.draw(need, rng);
        uint64_t b = bm;
        for (int i = 0; i < need; ++i) b |= card_bit(deck.cards[i]);
        const uint32_t hv = evaluate_mask(hm | b);
        int tied = 0;
        bool beaten = false;
        for (int o = 0; o < n_opp; ++o) {
            const uint32_t ov = evaluate_mask(om[o] | b);
            if (ov > hv) { beaten = true; break; }
            if (ov == hv) ++tied;
        }
        if (!beaten) total += 1.0 / (1.0 + tied);
    }
    return total / n_samples;
}

// Exact heads-up equity by enumerating every board completion.
// Returns (win, tie, lose) counts for hero.
std::array<uint64_t, 3> enumerate_equity_hu(const std::vector<int>& hero, const std::vector<int>& villain,
                                            const std::vector<int>& board) {
    if (hero.size() != 2 || villain.size() != 2) throw std::invalid_argument("both players need two cards");
    if (board.size() > 5) throw std::invalid_argument("board holds at most five cards");
    const uint64_t hm = mask_of(hero, "hero"), vm = mask_of(villain, "villain"), bm = mask_of(board, "board");
    if ((hm & vm) || (hm & bm) || (vm & bm)) throw std::invalid_argument("cards overlap");
    Deck deck;
    deck.build(hm | vm | bm);
    const int need = 5 - static_cast<int>(board.size());
    std::array<uint64_t, 3> out{0, 0, 0};
    auto score = [&](uint64_t b) {
        const uint32_t hv = evaluate_mask(hm | b), vv = evaluate_mask(vm | b);
        if (hv > vv) ++out[0];
        else if (hv == vv) ++out[1];
        else ++out[2];
    };
    const int n = deck.n;
    // Nested loops over combinations of `need` cards from the remaining deck.
    int idx[5];
    if (need == 0) { score(bm); return out; }
    for (int i = 0; i < need; ++i) idx[i] = i;
    while (true) {
        uint64_t b = bm;
        for (int i = 0; i < need; ++i) b |= card_bit(deck.cards[idx[i]]);
        score(b);
        int i = need - 1;
        while (i >= 0 && idx[i] == n - need + i) --i;
        if (i < 0) break;
        ++idx[i];
        for (int j = i + 1; j < need; ++j) idx[j] = idx[j - 1] + 1;
    }
    return out;
}

// Equity of several known hands against each other (exact when at most two
// cards are to come, Monte Carlo otherwise). Used by the simulator's
// hand-analysis tooling, not by the agent.
std::vector<double> equity_known_hands(const std::vector<std::vector<int>>& hands, const std::vector<int>& board,
                                       int n_samples, uint64_t seed) {
    const int n = static_cast<int>(hands.size());
    if (n < 2 || n > 10) throw std::invalid_argument("need between 2 and 10 hands");
    if (board.size() > 5) throw std::invalid_argument("board holds at most five cards");
    uint64_t known = mask_of(board, "board");
    std::vector<uint64_t> hm(n);
    for (int i = 0; i < n; ++i) {
        if (hands[i].size() != 2) throw std::invalid_argument("each hand needs two cards");
        hm[i] = mask_of(hands[i], "hand");
        if (hm[i] & known) throw std::invalid_argument("cards overlap");
        known |= hm[i];
    }
    Deck deck;
    deck.build(known);
    const uint64_t bm = mask_of(board, "board");
    const int need = 5 - static_cast<int>(board.size());
    std::vector<double> total(n, 0.0);
    std::vector<uint32_t> vals(n);
    double samples = 0.0;
    auto score = [&](uint64_t b) {
        uint32_t best = 0;
        for (int i = 0; i < n; ++i) { vals[i] = evaluate_mask(hm[i] | b); best = std::max(best, vals[i]); }
        int ties = 0;
        for (int i = 0; i < n; ++i) if (vals[i] == best) ++ties;
        for (int i = 0; i < n; ++i) if (vals[i] == best) total[i] += 1.0 / ties;
        samples += 1.0;
    };
    if (need <= 2 || n_samples <= 0) {
        const int m = deck.n;
        if (need == 0) score(bm);
        else if (need == 1) {
            for (int i = 0; i < m; ++i) score(bm | card_bit(deck.cards[i]));
        } else if (need == 2) {
            for (int i = 0; i < m; ++i)
                for (int j = i + 1; j < m; ++j) score(bm | card_bit(deck.cards[i]) | card_bit(deck.cards[j]));
        } else {
            // need in {3,4,5} with n_samples <= 0: full enumeration.
            int idx[5];
            for (int i = 0; i < need; ++i) idx[i] = i;
            while (true) {
                uint64_t b = bm;
                for (int i = 0; i < need; ++i) b |= card_bit(deck.cards[idx[i]]);
                score(b);
                int i = need - 1;
                while (i >= 0 && idx[i] == m - need + i) --i;
                if (i < 0) break;
                ++idx[i];
                for (int j = i + 1; j < need; ++j) idx[j] = idx[j - 1] + 1;
            }
        }
    } else {
        Rng rng(seed);
        for (int s = 0; s < n_samples; ++s) {
            deck.draw(need, rng);
            uint64_t b = bm;
            for (int i = 0; i < need; ++i) b |= card_bit(deck.cards[i]);
            score(b);
        }
    }
    for (double& t : total) t /= samples;
    return total;
}

// ---------------------------------------------------------------------------
// 169 preflop hand classes.
// ---------------------------------------------------------------------------
// Class index: pair -> r*13 + r; suited -> hi*13 + lo; offsuit -> lo*13 + hi.
inline int class_index(int c1, int c2) {
    const int r1 = c1 >> 2, r2 = c2 >> 2;
    const int hi = std::max(r1, r2), lo = std::min(r1, r2);
    if (hi == lo) return hi * 13 + hi;
    if ((c1 & 3) == (c2 & 3)) return hi * 13 + lo;
    return lo * 13 + hi;
}

struct ClassCombos {
    std::vector<std::vector<std::pair<int, int>>> combos;
    ClassCombos() : combos(169) {
        for (int a = 0; a < 52; ++a)
            for (int b = a + 1; b < 52; ++b) combos[class_index(a, b)].emplace_back(a, b);
    }
};
const ClassCombos CC;

// Number of non-conflicting (combo_a, combo_b) pairs for each class pair.
py::array_t<double> class_pair_weights() {
    py::array_t<double> out({169, 169});
    auto o = out.mutable_unchecked<2>();
    for (int a = 0; a < 169; ++a)
        for (int b = 0; b < 169; ++b) {
            double n = 0.0;
            for (const auto& x : CC.combos[a])
                for (const auto& y : CC.combos[b]) {
                    const uint64_t mx = card_bit(x.first) | card_bit(x.second);
                    const uint64_t my = card_bit(y.first) | card_bit(y.second);
                    if (!(mx & my)) n += 1.0;
                }
            o(a, b) = n;
        }
    return out;
}

// Monte Carlo class-vs-class preflop equity matrix (symmetric by construction).
py::array_t<float> class_equity_matrix(int n_samples, uint64_t seed) {
    if (n_samples < 1) throw std::invalid_argument("n_samples must be positive");
    py::array_t<float> out({169, 169});
    auto o = out.mutable_unchecked<2>();
    {
        py::gil_scoped_release release;
        Rng rng(seed);
        Deck deck;
        for (int a = 0; a < 169; ++a) {
            for (int b = a; b < 169; ++b) {
                const auto& ca = CC.combos[a];
                const auto& cb = CC.combos[b];
                double total = 0.0;
                int done = 0;
                for (int s = 0; s < n_samples; ++s) {
                    uint64_t ma, mb;
                    int tries = 0;
                    while (true) {
                        const auto& x = ca[rng.below(static_cast<uint32_t>(ca.size()))];
                        const auto& y = cb[rng.below(static_cast<uint32_t>(cb.size()))];
                        ma = card_bit(x.first) | card_bit(x.second);
                        mb = card_bit(y.first) | card_bit(y.second);
                        if (!(ma & mb)) break;
                        if (++tries > 1000) throw std::runtime_error("class pair has no consistent deal");
                    }
                    deck.build(ma | mb);
                    deck.draw(5, rng);
                    uint64_t bm = 0;
                    for (int i = 0; i < 5; ++i) bm |= card_bit(deck.cards[i]);
                    const uint32_t va = evaluate_mask(ma | bm), vb = evaluate_mask(mb | bm);
                    if (va > vb) total += 1.0;
                    else if (va == vb) total += 0.5;
                    ++done;
                }
                const float e = static_cast<float>(total / done);
                o(a, b) = e;
                o(b, a) = 1.0f - e;
            }
            o(a, a) = 0.5f;
        }
    }
    return out;
}

// Equity of each preflop class against n_opp random hands.
py::array_t<float> class_equity_vs_random(int n_opp, int n_samples, uint64_t seed) {
    if (n_opp < 1 || n_opp > 9) throw std::invalid_argument("n_opp must be in [1, 9]");
    if (n_samples < 1) throw std::invalid_argument("n_samples must be positive");
    py::array_t<float> out(169);
    auto o = out.mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        Rng rng(seed);
        Deck deck;
        for (int a = 0; a < 169; ++a) {
            const auto& ca = CC.combos[a];
            double total = 0.0;
            for (int s = 0; s < n_samples; ++s) {
                const auto& x = ca[rng.below(static_cast<uint32_t>(ca.size()))];
                const uint64_t hm = card_bit(x.first) | card_bit(x.second);
                deck.build(hm);
                deck.draw(5 + 2 * n_opp, rng);
                uint64_t b = 0;
                for (int i = 0; i < 5; ++i) b |= card_bit(deck.cards[i]);
                const uint32_t hv = evaluate_mask(hm | b);
                int tied = 0;
                bool beaten = false;
                for (int k = 0; k < n_opp; ++k) {
                    const uint64_t om = card_bit(deck.cards[5 + 2 * k]) | card_bit(deck.cards[6 + 2 * k]);
                    const uint32_t ov = evaluate_mask(om | b);
                    if (ov > hv) { beaten = true; break; }
                    if (ov == hv) ++tied;
                }
                if (!beaten) total += 1.0 / (1.0 + tied);
            }
            o(a) = static_cast<float>(total / n_samples);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Independent Chip Model (Malmuth-Harville) via subset dynamic programming.
// ---------------------------------------------------------------------------
std::vector<double> icm_equity(const std::vector<double>& stacks, const std::vector<double>& payouts) {
    const int n = static_cast<int>(stacks.size());
    if (n < 1 || n > 16) throw std::invalid_argument("icm supports 1 to 16 players");
    double total = 0.0;
    for (double s : stacks) {
        if (s < 0.0) throw std::invalid_argument("stacks must be non-negative");
        total += s;
    }
    const int places = std::min(static_cast<int>(payouts.size()), n);
    std::vector<double> ev(n, 0.0);
    if (places == 0 || total <= 0.0) return ev;
    const size_t m = static_cast<size_t>(1) << n;
    std::vector<double> f(m, 0.0), sum(m, 0.0);
    f[0] = 1.0;
    for (size_t mask = 1; mask < m; ++mask) {
        const int low = __builtin_ctzll(static_cast<unsigned long long>(mask));
        sum[mask] = sum[mask & (mask - 1)] + stacks[low];
    }
    for (size_t mask = 0; mask < m; ++mask) {
        const double fm = f[mask];
        if (fm == 0.0) continue;
        const int k = __builtin_popcountll(static_cast<unsigned long long>(mask));
        if (k >= places) continue;
        const double rem = total - sum[mask];
        if (rem <= 0.0) continue;
        const double pay = payouts[k];
        for (int i = 0; i < n; ++i) {
            if (mask & (static_cast<size_t>(1) << i)) continue;
            if (stacks[i] <= 0.0) continue;
            const double pr = fm * stacks[i] / rem;
            ev[i] += pr * pay;
            f[mask | (static_cast<size_t>(1) << i)] += pr;
        }
    }
    return ev;
}

// ---------------------------------------------------------------------------
// Exhaustive category census over all C(52,7) seven-card hands (test oracle).
// ---------------------------------------------------------------------------
std::vector<uint64_t> category_census7() {
    std::vector<uint64_t> counts(9, 0);
    py::gil_scoped_release release;
    uint64_t bits[52];
    for (int c = 0; c < 52; ++c) bits[c] = card_bit(c);
    for (int a = 0; a < 52; ++a) {
        const uint64_t ma = bits[a];
        for (int b = a + 1; b < 52; ++b) {
            const uint64_t mb = ma | bits[b];
            for (int c = b + 1; c < 52; ++c) {
                const uint64_t mc = mb | bits[c];
                for (int d = c + 1; d < 52; ++d) {
                    const uint64_t md = mc | bits[d];
                    for (int e = d + 1; e < 52; ++e) {
                        const uint64_t me = md | bits[e];
                        for (int f = e + 1; f < 52; ++f) {
                            const uint64_t mf = me | bits[f];
                            for (int g = f + 1; g < 52; ++g) {
                                ++counts[evaluate_mask(mf | bits[g]) >> CAT_SHIFT];
                            }
                        }
                    }
                }
            }
        }
    }
    return counts;
}

// ---------------------------------------------------------------------------
// Python bindings.
// ---------------------------------------------------------------------------
uint32_t evaluate_cards(const std::vector<int>& cards) {
    if (cards.size() < 5 || cards.size() > 7) throw std::invalid_argument("evaluate needs 5 to 7 cards");
    return evaluate_mask(mask_of(cards, "cards"));
}

uint32_t evaluate7(int a, int b, int c, int d, int e, int f, int g) {
    return evaluate_mask(card_bit(a) | card_bit(b) | card_bit(c) | card_bit(d) | card_bit(e) | card_bit(f) | card_bit(g));
}

py::array_t<uint32_t> evaluate_batch(const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& hands) {
    if (hands.ndim() != 2) throw std::invalid_argument("hands must have shape (N, k)");
    const py::ssize_t n = hands.shape(0), k = hands.shape(1);
    if (k < 5 || k > 7) throw std::invalid_argument("each hand needs 5 to 7 cards");
    py::array_t<uint32_t> out(n);
    auto h = hands.unchecked<2>();
    auto o = out.mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n; ++i) {
            uint64_t m = 0;
            for (py::ssize_t j = 0; j < k; ++j) m |= card_bit(h(i, j));
            o(i) = evaluate_mask(m);
        }
    }
    return out;
}

// Timing helper: evaluates `n` pseudo-random seven-card hands inside C++ and
// returns a checksum, so Python can measure raw evaluator throughput.
uint64_t evaluate_random7(int n, uint64_t seed) {
    uint64_t checksum = 0;
    py::gil_scoped_release release;
    Rng rng(seed);
    Deck deck;
    deck.build(0);
    for (int i = 0; i < n; ++i) {
        deck.draw(7, rng);
        uint64_t m = 0;
        for (int j = 0; j < 7; ++j) m |= card_bit(deck.cards[j]);
        checksum += evaluate_mask(m);
    }
    return checksum;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "microholdem C++ core: evaluator, Monte Carlo equity, ICM";
    m.attr("CAT_SHIFT") = CAT_SHIFT;
    m.def("evaluate", &evaluate_cards, py::arg("cards"), "Evaluate 5 to 7 cards (list of ints 0..51).");
    m.def("evaluate7", &evaluate7, "Evaluate seven cards given positionally.");
    m.def("evaluate_mask", [](uint64_t mask) { return evaluate_mask(mask); }, py::arg("mask"));
    m.def("evaluate_batch", &evaluate_batch, py::arg("hands"), "Evaluate an (N, k) int32 array of hands.");
    m.def("evaluate_random7", &evaluate_random7, py::arg("n"), py::arg("seed"));
    m.def("card_mask", [](const std::vector<int>& cards) { return mask_of(cards, "cards"); }, py::arg("cards"));
    m.def("equity_vs_random", &equity_vs_random, py::arg("hero"), py::arg("board"), py::arg("n_opp"),
          py::arg("n_samples"), py::arg("seed"));
    m.def(
        "equity_vs_ranges",
        [](const std::vector<int>& hero, const std::vector<int>& board,
           const std::vector<std::pair<py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
                                       py::array_t<double, py::array::c_style | py::array::forcecast>>>& ranges,
           int n_samples, uint64_t seed) {
            std::vector<WeightedRange> rs;
            rs.reserve(ranges.size());
            for (const auto& r : ranges) rs.push_back(make_range(r.first, r.second));
            return equity_vs_ranges(hero, board, rs, n_samples, seed);
        },
        py::arg("hero"), py::arg("board"), py::arg("ranges"), py::arg("n_samples"), py::arg("seed"));
    py::class_<WeightedRange>(m, "CompiledRange", "A weighted set of hole-card combos compiled for sampling.")
        .def(py::init([](const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& combos,
                         const py::array_t<double, py::array::c_style | py::array::forcecast>& weights) {
                 return make_range(combos, weights);
             }),
             py::arg("combos"), py::arg("weights"))
        .def_property_readonly("size", [](const WeightedRange& r) { return r.c1.size(); });
    m.def(
        "equity_vs_compiled",
        [](const std::vector<int>& hero, const std::vector<int>& board, const std::vector<WeightedRange>& ranges,
           int n_samples, uint64_t seed) { return equity_vs_ranges(hero, board, ranges, n_samples, seed); },
        py::arg("hero"), py::arg("board"), py::arg("ranges"), py::arg("n_samples"), py::arg("seed"),
        "Equity against pre-compiled ranges (no per-call range parsing).");
    m.def("enumerate_equity_hu", &enumerate_equity_hu, py::arg("hero"), py::arg("villain"), py::arg("board"));
    m.def("equity_known_hands", &equity_known_hands, py::arg("hands"), py::arg("board"), py::arg("n_samples"),
          py::arg("seed"));
    m.def("class_index", &class_index, py::arg("c1"), py::arg("c2"));
    m.def("class_pair_weights", &class_pair_weights);
    m.def("class_equity_matrix", &class_equity_matrix, py::arg("n_samples"), py::arg("seed"));
    m.def("class_equity_vs_random", &class_equity_vs_random, py::arg("n_opp"), py::arg("n_samples"), py::arg("seed"));
    m.def("icm_equity", &icm_equity, py::arg("stacks"), py::arg("payouts"));
    m.def("category_census7", &category_census7);
}
