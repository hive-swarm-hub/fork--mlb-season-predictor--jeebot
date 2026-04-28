# Season Projection Priors

These are candidate priors for MLB season-long standings forecasts. They are intentionally separate from `program.md` so agents can decide which knowledge to use.

## Projection Aggregation

Multi-source player projections are usually stronger than a single projection source. Candidate features include Steamer WAR, ZiPS WAR, Depth Charts WAR, and their player-level blend. Breakdowns by position-player WAR, starting-pitcher WAR, relief WAR, and high-leverage relief WAR may behave differently.

## Underlying Skill Indicators

For hitters, xwOBA, barrel rate, hard-hit rate, exit velocity, and chase rate can be more stable than outcome stats like batting average or wOBA. For pitchers, stuff+ and location+ can preserve signal that ERA and FIP miss.

## Roster Construction

Rotation depth matters because teams usually need more than five starters. SP1-SP7 quality, SP4-SP7 falloff, bullpen leverage, projected playing time, and call-up depth can all affect season variance.

## Catching And Defense

Catcher framing can be worth meaningful runs over a full season. DRS, UZR, and OAA can disagree, so blended defense may be more robust than any single defensive metric.

## Team Context

Pythagorean record, BaseRuns record, and third-order winning percentage often carry more signal than last year's actual win-loss record. Schedule strength, intra-division schedule strength, and multi-year park factors also matter.

## Aging And Durability

Hitters past roughly 29 and pitchers past roughly 28 often need stronger regression. Players in the 24-27 range may deserve different priors. Durability changes both expected wins and variance: a reliable 200-inning starter and an equally talented 80-inning starter are not equivalent.

## Uncertainty

Projected win totals should usually be distributions, not just point estimates. Teams relying on prospects, injured players, fragile rotations, or volatile bullpens may have wider intervals even at the same mean.