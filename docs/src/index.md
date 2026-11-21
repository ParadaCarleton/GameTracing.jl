# GameTracing

Homotopy continuation for logit quantal response equilibria. `trace_endpoint` follows a logit game's branch from
zero precision (infinite temperature), where the only equilibrium is the flat one, up to a target precision. It crosses
folds by pseudo-arclength continuation, on BifurcationKit (`BifurcationKitSettings`) or on lib/PathFollowing
(`PathFollowingSettings`). Only BifurcationKit redraws the game mid-trace.
