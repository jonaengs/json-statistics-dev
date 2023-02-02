
## UTILS
- [x] Make logger put old files in a separate folder

## PRUNING
- [x] Implement max no. paths pruning  
- [x] Support for selecting and combining multiple strategies + doing so with cmd args.
- [ ] Implement suffix and prefix key path pruning
- [ ] Make something that can use statistics with (suffix/prefix)-pruned key paths

## STATS
- [x] Find some way to test for "key-path type confusion"
- [ ] Explicitly differentiate between equi-width and singleton histograms. Update code to reflect this
- [ ] Check assumptions around sampling and what we can say about min, max, ndv and histogram stats and how we use them in estimation

## ANALYSIS
- [x] Find all key-paths, both typed and untyped
- [x] Find the value range for all key-paths
  - [ ] Maybe: Store ground-truth statistics (no sampling, no prob. dats structures) to do this
- [x] Find some strategy for picking values to test with all key-paths
  - [ ] Find an improved approach
- [ ] Find some way to test various performance metrics for all key-paths with those test values
- [ ] Visualize the results
- [ ] Analyze the result

## TESTING
Write tests for all important logic (wasting time on erroneous results sucks)
- [x] Test all estimates with basic stats
- [x] Test all estimates with basic_ndv stats
- [x] Test all estimates with hyperloglog stats
- [ ] Test all estimates with histogram stats