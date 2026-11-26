# microholdem on Chipzen

`bot.py` runs microholdem as a [Chipzen](https://chipzen.ai) bot, built on the
[Chipzen SDK](https://github.com/chipzen-ai/chipzen-sdk) (`chipzen-bot` 0.7.0). For every
`turn_request` it rebuilds the `Observation` microholdem's own engine would produce: per-seat stacks
and street bets, folds, busted seats, blinds and the action history, all replayed from the hand's
`action_history`. `test_bot.py` checks that rebuild against every decision point of 400 hands played
by microholdem's engine at 2 to 6 seats, including short stacks and all-ins.

The agent is `microholdem.headsup.HeadsUpAgent`: at two-seat tables it keeps a frequency model of
the opponent, fed from every `round_result`, and at larger tables it plays as `MicroAgent`.

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

## Measured against the deployed version

`arena.py` plays this format locally: microholdem's engine deals, and each `chipzen.Bot` gets the
`match_start`, `turn_request` and `round_result` payloads the platform sends. Matches are mirrored
pairs (same seed, seats swapped) with a fixed 600-sample Monte Carlo and seeded agents, so a run
depends only on the seed and repeats exactly. `v1` is the adapter as first uploaded (`git show b5420dd:integrations/chipzen/bot.py`).

| Opponent | v1 (deployed) | This version | Matches each |
|---|---:|---:|---:|
| v1, head to head | | 68.3% (95% CI 66.2 to 70.3) | 2,000 |
| v1 with only the two fixes above, head to head | | 56.8% (54.6 to 59.0) | 2,000 |
| minraiser (opens 80% for the minimum, bets half the time; after 6BP9) | 42.7% | 60.6% | 1,000 |
| barreler (opens or limps 96%, re-raises 19%, shoves 8%; after Phineas-IV) | 59.4% | 67.3% | 1,000 |
| station (limps everything, rarely folds; after glambot) | 77.2% | 96.6% | 1,000 |
| regular (opens 72%, re-raises 12%) | 64.2% | 76.6% | 1,000 |
| sticky (never folds before the flop, calls a shove with 45% of hands; after Chatterbox) | 61.4% | 80.7% | 1,000 |
| oracle, granite, jammer, sponge (microholdem archetypes) | 68.6, 86.4, 65.6, 71.1% | 90.8, 95.0, 78.8, 94.3% | 1,000 |
| blitz (raises most hands, shoves 15% of its raises) | 63.9% | 57.3% | 1,000 |
| All ten pool opponents | 66.1% | 79.8% | 10,000 |

Reproduce (about 13 minutes on eight cores):

```bash
git show b5420dd:integrations/chipzen/bot.py > /tmp/bot_v1.py
python arena.py --a bot.py --b /tmp/bot_v1.py --matches 2000
python arena.py --a bot.py --pool --matches 1000
python arena.py --a /tmp/bot_v1.py --pool --matches 1000
```

What this does and does not show. The pool opponents are simple bots written for this benchmark,
five of them shaped on frequencies read from replays; they are not the bots on the platform, and a
win rate against them is not a forecast of a ladder rating. The version is worse against blitz,
the one opponent that raises and shoves almost at random. The head-to-head result is the firm one:
the same engine, the same seeds, and only the adapter and agent changed.

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
False, which is the old behaviour, it wins 50.4% of 2,000 matches (95% CI 48.2 to 52.6), and its pool
results moved within noise. The spot is rare, and four of the five profile opponents fold all but the
top 6% of hands to a shove, against which shoving everything is the best play. The change matters
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
