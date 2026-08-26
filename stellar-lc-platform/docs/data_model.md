# Initial Data Model

Likely core entities:

- `Star`
- `Identifier`
- `Observation`
- `LightCurveProduct`
- `QCResult`
- `DetrendingRun`
- `PeriodRun`
- `ConsensusPeriod`
- `StellarProperties`
- `AnalysisRun`

Each derived result should eventually preserve:

- input identifier / file
- method
- parameter values
- code version
- timestamp
- uncertainty
- quality/confidence metrics
- output artifact locations
