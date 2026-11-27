# microholdem on Chipzen

`bot.py` runs microholdem as a [Chipzen](https://chipzen.ai) bot, built on the
[Chipzen SDK](https://github.com/chipzen-ai/chipzen-sdk) (`chipzen-bot` 0.7.0). For every
`turn_request` it rebuilds the `Observation` microholdem's own engine would produce: per-seat stacks
and street bets, folds, busted seats, blinds and the action history, all replayed from the hand's
`action_history`. `test_bot.py` checks that rebuild against every decision point of 400 hands played
by microholdem's engine at 2 to 6 seats, including short stacks and all-ins.

The agent is `microholdem.headsup.HeadsUpAgent`: at two-seat tables it keeps a model of the opponent,
fed from the actions in every `round_result` and from the cards shown at showdown, and at larger
tables it plays as `MicroAgent`.

## Match format and what the first version got wrong

A Chipzen bot-vs-bot match is a heads-up freezeout: 10,000 chips each, blinds from 50/100 rising
every 20 hands (75/150, 100/200, 150/300, 200/400, 300/600 seen so far), played until one seat has
every chip. `match_start.game_config` names only the opening blinds.

The first version of this adapter took the blinds from `match_start` for the whole match. From hand
21 on it played as if the blinds were still 50/100: it folded its big blind to 78% of small opens
(24% at the first level), sized opens to the minimum and never reached its push/fold tables. In its
first 11 matches on the platform it made +252 BB/100 at the first level and +28 BB/100 after it.
The blinds are now read from each hand's `post_small_blind` and `post_big_blind` entries, as
Chipzen's `COMMON-PITFALLS.md` section 11 says to.

It also priced calls against a covering all-in on the whole shove, not on the part it could call,
and so called off too light. `microholdem.headsup.callable_view` fixes that.

## Measured against the earlier versions

`arena.py` plays this format locally: microholdem's engine deals, and each `chipzen.Bot` gets the
`match_start`, `turn_request` and `round_result` payloads the platform sends. Matches are mirrored
pairs (same seed, seats swapped) with a fixed 600-sample Monte Carlo and seeded agents, so a run
depends only on the seed and repeats exactly. `v1` is the adapter as first uploaded
(`b5420dd`); `previous` is the version before the postflop change described below (`d6afb2b`).

| Opponent | v1 (deployed) | Previous | This version | Matches each |
|---|---:|---:|---:|---:|
| v1, head to head | | 68.3% | 66.2% (95% CI 64.1 to 68.3) | 2,000 |
| v1 with only its two bugs fixed, head to head | | 56.8% | 55.3% (53.1 to 57.5) | 2,000 |
| previous, head to head | | | 55.3% (53.1 to 57.5) | 2,000 |
| minraiser (opens 80% for the minimum, bets half the time; after 6BP9) | 42.7% | 60.6% | 59.8% | 1,000 |
| barreler (opens or limps 96%, re-raises 19%, shoves 8%; after Phineas-IV) | 59.4% | 67.3% | 67.4% | 1,000 |
| station (limps everything, rarely folds; after glambot) | 77.2% | 96.6% | 93.2% | 1,000 |
| regular (opens 72%, re-raises 12%) | 64.2% | 76.6% | 75.3% | 1,000 |
| sticky (never folds before the flop, calls a shove with 45% of hands; after Chatterbox) | 61.4% | 80.7% | 81.9% | 1,000 |
| honest (bets the pot only with a strong hand, 16% of its chances) | 64.9% | 87.9% | 86.7% | 1,000 |
| bully (bets the pot 26% of its chances, two times in three without a strong hand) | 66.4% | 77.7% | 78.9% | 1,000 |
| oracle, granite, jammer, sponge (microholdem archetypes) | 68.6, 86.4, 65.6, 71.1% | 90.8, 95.0, 78.8, 94.3% | 86.8, 91.9, 77.3, 90.8% | 1,000 |
| blitz (raises most hands, shoves 15% of its raises) | 63.9% | 57.3% | 66.6% | 1,000 |
| All twelve pool opponents | 66.0% | 80.3% | 79.7% | 12,000 |

Reproduce (about 25 minutes on eight cores). The previous version needs its own copy of the agent
module, because both versions are loaded in one process:

```bash
mkdir -p /tmp/prev
git show b5420dd:integrations/chipzen/bot.py > /tmp/prev/bot_v1.py
git show d6afb2b:src/microholdem/headsup.py \
  | sed -e 's/^from \. import/from microholdem import/' -e 's/^from \.\([a-z]*\) import/from microholdem.\1 import/' \
  > /tmp/prev/headsup_prev.py
git show d6afb2b:integrations/chipzen/bot.py \
  | sed 's/^from microholdem.headsup import/from headsup_prev import/' > /tmp/prev/bot_prev.py
export PYTHONPATH=/tmp/prev
python arena.py --a bot.py --b /tmp/prev/bot_prev.py --b /tmp/prev/bot_v1.py --matches 2000
python arena.py --a bot.py --pool --matches 1000
python arena.py --a /tmp/prev/bot_prev.py --pool --matches 1000
python arena.py --a /tmp/prev/bot_v1.py --pool --matches 1000
python arena.py --a bot.py --base /tmp/prev/bot_prev.py --pool --chips 300    # the table in the next section
```

What this does and does not show. The pool opponents are simple bots written for this benchmark,
five of them shaped on frequencies read from replays and two built as a pair of rare bettors with
opposite hands; they are not the bots on the platform, and a win rate against them is not a
forecast of a ladder rating. The head-to-head results are the firm ones: the same engine, the same
seeds, and only the adapter and agent changed. Against the pool this version wins no more matches than the previous one. It wins
more against the opponents that bet most and 3 to 4 points fewer against station, oracle, granite
and sponge, which the previous version already beat nine times in ten or more; the next section
says why.

## Reading a bet

The previous version judged a bet on a 50-cell grid of hand strength: its equity against the
opponent's whole range was taken as the share of that range it beat, and a bet removed most of the
weaker cells. Logged against the opponent's real cards over 16,000 bets faced across the twelve pool
opponents, that estimate averaged 25% where the bot in fact won 35% of the time, and 24% against
36% on the flop. A draw was the worst case: seven-high with a flush draw was read like seven-high.
So the bot folded to bets it had the odds to call. In its first rated match the new upload folded
to 20 of 24 postflop bets and raises.

This version weights every hand the opponent can hold by what it did this hand, ordering the hands
by what they make on each street's board, and takes its equity against those weighted hands with
the cards still to come. On the same 16,000 bets that estimate averages 33% (36% on the flop). It
folds to 57% of them where the old read would have folded to 72%. It then counts each bet and raise
whose hand is shown at showdown by how strong that hand was, per fifth of the range, and moves the
assumed content of the next bet towards what this opponent has shown.

Match counts separate two close versions slowly, so the change was measured in chips per hand:
sessions of 120 hands, 20 at each of the first six blind levels, with both stacks back at 10,000
before every hand and the opponent model started fresh. Both versions get the same cards, and each
session is played from both seats. Big blinds won per 100 hands over 300 sessions (72,000 hands)
per opponent, with the standard error of the paired difference:

| Opponent | Previous | This version, difference |
|---|---:|---:|
| minraiser | 16.1 | +10.6 ± 2.3 |
| barreler | 43.7 | +18.7 ± 2.4 |
| station | 327.7 | +21.6 ± 2.5 |
| regular | 72.7 | +13.9 ± 2.5 |
| sticky | 114.6 | +16.0 ± 2.5 |
| honest | 134.1 | +15.8 ± 2.4 |
| bully | 99.1 | +25.5 ± 3.2 |
| oracle | 155.2 | +26.6 ± 3.1 |
| blitz | 81.5 | +97.2 ± 6.5 |
| granite | 44.7 | -0.8 ± 0.9 |
| jammer | 56.3 | +3.2 ± 1.8 |
| sponge | 253.1 | +8.9 ± 1.7 |
| All twelve | 116.6 | +21.4 ± 1.0 |

More chips per hand did not become more matches against the weakest opponents. Calling more bets
makes pots larger, and in a freezeout a clear favourite wins less often when the swings grow faster
than its edge: against oracle and granite, at 50 to 100 big blinds deep, the chips won per hand
rose while the ratio of chips won to their variance fell by about a quarter. Against an equal opponent only the chips matter, which is the
55.3% head to head.

What it still does not do:

- **Tell a rare bluffer from a rare value bettor quickly.** Against honest and bully, which bet 16%
  and 26% of the time they can, it folds to 81% and 71% of bets. A match gives it one or two of
  such a seat's bets at showdown, too few to move the read far.
- **Keep a floor under its calling.** Calling a fixed share of the minimum defence frequency until
  showdowns say otherwise was tried in five forms. Each gained against bully and blitz and lost
  against opponents that only bet strong hands (up to 23 big blinds per 100 hands against sponge),
  and in 109 Season 16 replays the bots that bet under 30% of the time showed a weak hand in 18% of
  their shown bets. It is not in this version.
- **Read the river as well as the flop.** Facing a river bet its estimate averages 26% against a
  true 33%.

The starting assumptions were checked against those replays, not against the pool: the field bets
38% of the time it can, 29% of its shown bets were made with less than even equity against a random
hand, five shown raises in six were made with more than 65%, and hands above 80% were checked 38%
of the time they could have bet. The counts over bets faced, the variance ratio and the replay
figures are one-off measurements; the two tables above are reproducible with the commands given.

## Short stacks against an open

The first upload of this version lost the last hand of its first rated match by shoving eight-seven
offsuit for 17 big blinds over a 2.15x open. With 20 big blinds or fewer, `MicroAgent` prices a reply
to a small open as a call of that open and then shoves the whole stack, so it shoves every hand. The
first adapter never reached that code, because it believed the blinds were still 50/100.

`HeadsUpAgent` now values folding, calling and shoving in chips and plays the best of the three. A
shove is valued against how often the opener folds to one. With no shoves seen, that is the fold rate
of the shove-or-not game solved for that stack and that opening range (61% at 20 big blinds, 33% at
10); every shove the opponent answers moves it towards the opponent's own rate. A fresh agent shoves
about a third of its hands at 20 big blinds and 44% at 10.

This is not a gain in the arena. Against the same bot with `HeadsUpAgent._is_short_reply` returning
False, which is the old behaviour, it won 50.4% of 2,000 matches (95% CI 48.2 to 52.6) when it was
added, and its pool results moved within noise. The spot is rare, and most of the profile opponents
fold all but the top 6% of hands to a shove, against which shoving everything is the best play. The change matters
against opponents that call: there the old line loses the stack as an underdog, and this one does not.

## Differences from microholdem's own engine

| Setting | microholdem default | On Chipzen |
|---|---|---|
| Decision time | 1 ms per action | 5 s per action (hosted); the agent uses up to 20,000 Monte Carlo samples or 20 ms |
| Payouts | ICM when a payout structure is given | Chip EV: matches carry no payout structure |
| Antes | Part of the preflop bet | Dead money, as the Chipzen protocol defines them |
| Table size | 2 to 10 seats | 2 to 6 seats |

## Check it locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ../.. chipzen-bot==0.7.0
python -m pytest -q test_bot.py
chipzen-sdk validate . --check-connectivity
```

`validate` runs the upload pipeline's checks plus a protocol exchange against an in-process mock
server. It does not measure strength; ratings come from matches on the platform.

## Option A: upload a Docker image (hosted play)

Needs a Docker engine. From the repository root:

```bash
docker build --platform=linux/amd64 -f integrations/chipzen/Dockerfile -t microholdem-chipzen:2 .
docker run --rm --platform=linux/amd64 --entrypoint id microholdem-chipzen:2   # must not print uid=0
docker image ls microholdem-chipzen:2                                           # image limit: 200 MB
docker save microholdem-chipzen:2 | gzip > integrations/chipzen/microholdem-chipzen-v2.tar.gz  # upload limit: 250 MB; git-ignored
```

Upload the tarball in the Chipzen developer UI. Review plays a short scripted match and activates the
bot when it passes. The platform runs images on linux/amd64 with a read-only filesystem, seccomp,
0.5 vCPU and 256 MB; microholdem writes nothing at runtime and starts in well under a second.

## Option B: remote play from your own machine

No Docker needed, but the machine must stay online while matches run.

1. Create a remote (external-API) bot on chipzen.ai and copy its bot id and its `cz_extbot_` token.
2. Create `chipzen.toml` in this folder. It is git-ignored; never commit the token.

   ```toml
   [external_api]
   token = "cz_extbot_..."
   bot_id = "<bot uuid>"
   ```

3. Run `chipzen run-external bot.py`.

## Notes

- An uploaded image cannot be replaced: uploading a new tarball creates a new bot, and a season
  entry stays with the bot that was registered.
- The first image passed the platform's review and played its matches there, so the C++ extension
  runs under the platform's seccomp profile.
