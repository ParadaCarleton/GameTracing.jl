# Games the tests and the arm benchmark share.

# LogitNash.jl's games with two actions per player (test/test_e2e.jl and its README), as integer payoff tables.
const LOGITNASH_GAMES = (
    battle_of_the_sexes = ([3 0; 0 2], [2 0; 0 3]),
    coordination_wiki = ([2 1; 1 2], [4 3; 3 4]),
    coordination_pure = ([8 0; 0 8], [8 0; 0 8]),
    coordination_assurance = ([8 0; 0 5], [8 0; 0 5]),
    stag_hunt = ([8 0; 7 5], [8 7; 0 5]),
    prisoners_dilemma = ([5 0; 8 1], [5 8; 0 1]),
    matching_pennies = ([1 -1; -1 1], [-1 1; 1 -1]),
    three_prisoners = (
        cat([3 1; 5 2], [1 0; 2 1]; dims = 3),
        cat([3 5; 1 2], [1 2; 0 1]; dims = 3),
        cat([3 1; 1 0], [5 2; 2 1]; dims = 3),
    ),
)

# Games symmetric under relabeling actions (battle of the sexes also swaps players): the principal branch meets an
# exact pitchfork (battle of the sexes at precision 0.808), past which it is ambiguous. Exact symmetry is measure
# zero, so these games assert only that the endpoint is an equilibrium of the map.
const PITCHFORK_GAMES = (:battle_of_the_sexes, :coordination_wiki, :coordination_pure)

# Three players, two actions each. The principal branch folds back in precision between 31//10 and 32//10, so a
# corrector started from a predictor past the target can converge onto the folded-back segment.
const OVERSHOT_GAME = (
    [358 -73; -175 -10;;; 113 -822; -685 390] .// 100,
    [183 -129; 53 -357;;; -158 235; -354 -285] .// 100,
    [-175 -405; 87 -124;;; 10 -177; -110 180] .// 100,
)

# Three players, two actions each (a random draw, rounded to 1e-6). Its principal branch folds at precision about 3.3617.
# Gambit's default steps round that fold in one step, from 3.349675 to 3.128675, so `logit_solve_lambda` misses the first
# crossing of any precision between them and returns a later one, near the pure profile (1, 2, 2).
const FOLDED_GAMBIT_GAME = (
    [-4108184 -6199065; -140340 -5617899;;; 493257 3009713; 540823 127869] .// 10^6,
    [-359879 -9625560; 3795891 -3394274;;; -3946851 1662104; 3214943 2863915] .// 10^6,
    [-503395 -5503689; -4888492 2357340;;; -1217139 -3504025; -3129362 207368] .// 10^6,
)

# Three players of 2, 3 and 2 actions (seed 0x7d771e5515778fcd of the logit_solve_lambda agreement testset, rounded to
# 1e-6). Its principal branch folds just past precision 1. Gambit's default steps round that fold, from 0.9992 to
# 0.9743, so `logit_solve_lambda` at precision 1 returns a later crossing (CI run 44506024991).
const FOLD_PAST_ONE_GAME = (
    [1020534 -4195093 -3847093; 2187863 -4056740 3647491;;; -935116 1005626 -1354133; -4185190 3055487 -5356626] .// 10^6,
    [6041366 -705927 90218; 2650821 3632445 4345817;;; 3410121 -515812 2551517; -49347 -3885990 1378177] .// 10^6,
    [-2799332 3279993 -411480; -2686956 -3254749 1100361;;; -6313397 5003338 2269424; -1033173 -1135836 -1219361] .// 10^6,
)

# Two players, three actions each (a random draw, rounded to 1e-6). Gambit's branch takes 926199 points and 172 s on
# macdev to reach regret 1e-12, near λ 3.5e10; 41 points reach precision 4.
const SLOW_REGRET_GAME = (
    [-5025192 -3796184 96451; -2017419 -290936 1101438; -1299614 522459 -2982700] .// 10^6,
    [2802902 -2338006 -1788818; 1527677 -2060137 -303028; -2687319 805313 2589683] .// 10^6,
)

# Four players, two actions each (seed 0x2c06c04a5d81b3dc of the random-game testset, rounded to 1e-6). At precision
# about 3.05, Gambit leaves its principal branch, ending at first choices (0.16, 0.36, 0.08, 0.14) (CI run 41370200196).
const GAMBIT_LEAVES_GAME = (
    [1951724 1176029; 1415337 2765348;;; 998115 1201523; 2814368 -7103004;;;; -1878885 1184060; 5947423 -6109252;;; 1673502 -1027796; -888519 2955579] .// 10^6,
    [-2486602 566455; -2904504 -3764541;;; -5818249 30625; 2485436 618336;;;; 2033055 342635; -2769907 -1540549;;; -1633480 -2665809; 125757 659713] .// 10^6,
    [-2096866 1302905; 999183 2776673;;; -569484 -3032630; -4382120 7834803;;;; 1660815 -9421362; 2953784 3122954;;; -1815115 4488691; 2488710 2540069] .// 10^6,
    [-207281 1181961; 4483016 2010697;;; 295707 -1804204; 3555830 2867891;;;; 2785242 7372728; 2213823 994217;;; -3226846 3021391; 573187 5247358] .// 10^6,
)

# The battle of the sexes perturbed by about 1e-4. Near precision 0.808 its principal branch turns steeply (dλ/ds about
# 0.0024) past the middle arm of the unfolded pitchfork, and ends near the pure equilibrium (1, 1).
const UNFOLDED_PITCHFORK_GAME = ([30000726 -37; 757 20000821] .// 10^7, [19999477 -283; 919 29999523] .// 10^7)

# The (3, 3, 2) game of `3 .* randn(Xoshiro(152), Float32, (3, 3, 2))` per player. A long early PathFollowing step lands
# where its oriented tangent runs back along the branch; without the orientation check the trace ends at another
# equilibrium, 0.9995 away in probability from Gambit's pure one at λ ≈ 3.8.
const REVERSAL_GAME = (
    cat(
        Float32[-0.5337934f0 -3.4574738f0 -0.31516188f0; 1.2419287f0 0.6020585f0 -0.45870376f0; 1.6466057f0 -1.8631278f0 -0.5691585f0],
        Float32[-0.11642508f0 1.8486631f0 1.6131696f0; -4.5490217f0 -1.7635972f0 -4.980587f0; 1.3529017f0 -3.944703f0 -3.9721947f0];
        dims = 3,
    ),
    cat(
        Float32[5.1817837f0 4.2066636f0 -1.6429582f0; 0.27198213f0 3.56285f0 -2.5637565f0; 1.1680062f0 -2.4406242f0 -2.0018725f0],
        Float32[-3.496707f0 0.9063864f0 -1.136579f0; 0.5411101f0 0.86526525f0 -0.27381507f0; 7.237438f0 -3.9687374f0 2.3237944f0];
        dims = 3,
    ),
    cat(
        Float32[-0.90711546f0 1.2572101f0 2.234933f0; 0.7166976f0 -3.4552052f0 -0.39850372f0; 1.7495403f0 -2.8394732f0 -3.8426185f0],
        Float32[-0.48643172f0 5.1313787f0 0.56367534f0; 0.40493193f0 4.1448507f0 0.9289435f0; 4.757027f0 0.33046308f0 -0.8036462f0];
        dims = 3,
    ),
)

# One logit whose branch folds back at precision about 1.4; target 2 lies past the fold.
const FOLD_GAME = ThresholdPass(6.0f0, 3.0f0)

# A random 3x3 draw whose principal branch folds back at precision about 21.57, turns again near 2.5, then rises.
# Float32 BifurcationKit at the Gambit settings stalled at the first fold under per-logit Bernoulli weights.
const FOLDING_3X3_GAME = (
    Float32[0.8496506f0 -5.1560864f0 -5.0248613f0; -0.8010247f0 4.4318876f0 -5.6464076f0; 0.81554914f0 -3.3132758f0 -1.5674677f0],
    Float32[-4.9134245f0 -2.9423323f0 1.9437922f0; -4.345395f0 -1.6042254f0 0.86286294f0; 3.2346768f0 1.2498221f0 -0.47711927f0],
)

# A 2x2x2 draw of `logit_solve_tests.jl`'s random games, whose principal branch folds back near 23.27 and turns near 3.35.
# Float32 BifurcationKit at the Gambit settings stalled at the first fold under per-logit Bernoulli weights.
const FOLDING_2X2X2_GAME = (
    Float32[0.45097172f0 2.1065488f0; 0.25923342f0 5.8263507f0;;; 1.4212247f0 -3.3518772f0; -1.676609f0 -2.8861334f0],
    Float32[-3.0872922f0 -4.762417f0; 0.25956222f0 0.17571005f0;;; -0.20159242f0 0.77131903f0; 2.1493049f0 -4.9243317f0],
    Float32[2.896038f0 -3.6309934f0; 2.4591706f0 4.723743f0;;; -4.216482f0 -4.0223656f0; 0.52297986f0 2.3564496f0],
)
