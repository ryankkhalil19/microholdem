# microholdem on Chipzen

`bot.py` runs microholdem's `MicroAgent` as a [Chipzen](https://chipzen.ai) bot, built on the
[Chipzen SDK](https://github.com/chipzen-ai/chipzen-sdk) (`chipzen-bot` 0.7.0). For every
`turn_request` it rebuilds the `Observation` microholdem's own engine would produce: per-seat stacks
and street bets, folds, busted seats, blinds and the action history, all replayed from the hand's
`action_history`. `test_bot.py` checks that rebuild against every decision point of 400 hands played
by microholdem's engine at 2 to 6 seats, including short stacks and all-ins.

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
docker build --platform=linux/amd64 -f integrations/chipzen/Dockerfile -t microholdem-chipzen:1 .
docker run --rm --platform=linux/amd64 --entrypoint id microholdem-chipzen:1   # must not print uid=0
docker image ls microholdem-chipzen:1                                           # image limit: 200 MB
docker save microholdem-chipzen:1 | gzip > integrations/chipzen/microholdem-chipzen.tar.gz  # upload limit: 250 MB; git-ignored
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

## Not yet verified

- The hosted image has not been built or run here. If it fails at startup, the usual cause is the
  platform's seccomp profile blocking a syscall that a native extension needs (Chipzen DEV-MANUAL §7.4).
