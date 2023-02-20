
## GENERAL
- [ ] Separate int and float everywhere. When querying for numbers, query for both types (except for equality?) 
  * JSON Tiles differentiates between floats and ints. We can solve the uncertainty of whether the key-path can lead to the other type due to examples being skipped during sampling by combining an estimate for both.
  * How do we combine those estimates? Maybe we can use the skip_threshold times the heuristic multiplier and add that to the estimate we have. 
  * If we have seen lots of one type and not enough of the other, reduce the estimate if the other is high? Because it seems less likely that we would see so few of the other type if we've seen many of this one?
- [x] For strings: If there are loads of strings, but a small number of them are very common, maybe make a singleton histogram for those? This will both increase accuracy for those select strings, but also for the remaining strings as their estimate will be lower. 
  * E.g., If a key-path occurs 1k times, leading to unique 200 strings, where 5 of them occur > 100 times each, then we put those five in a special case singleton histogram. The remainder then have their estimated cardinality halved (instead of 1000/200, it becomes 500/195)
  * Special case as in: It will have to note that values were omitted, so we don't assume cardinality 0 if a lookup value is not in the histogram.
- [x] Store most-common-value? For certain fields, this could be very useful? (String fields where the empty string dominates). Like with the above singleton-ish histogram, it would improve accuracy for both the common value as well as the remaining values. 
  - [ ] Or store top-k values? Linear-ish computation. Not too expensive if we're already finding min and max, especially if we do all of them in a single pass. 
- [ ] Identify date strings?
- [x] Handle lists of enums
  * Once we have these enum array histograms, can we delete the histograms for each array child element?
  * We would need some threshold? enum array length, value frequencies, etc.
- [ ] Make random happen once at the start of the program, so sampling etc. is also deterministic
- [ ] Identify and handle pairs/tuples? 
- [ ] Should singleton histograms be maps instead of lists? Especially when so large. 
- [ ] Support JSON_CONTAINS_PATH()? (https://dev.mysql.com/doc/refman/8.0/en/json-search-functions.html#function_json-contains-path)


## BUGS
- [ ] isinstace(True, int) => True because bool subclasses int. Make sure this does not cause bugs anywhere. 

## UTILS / TRACKERS / LOGGER / CACHING
- [x] Make logger put old files in a separate folder
- [ ] Make tracker decorators take an optional argument specifying extra information to identify the function by so that we can differentiate between e.g., the time a function uses when different StatTypes are active.
- [ ] Rewrite settings to be something like a nested class or dataclass (but singleton?) , for better autocomplete and refactoring

## PRUNING
- [x] Implement max no. paths pruning  
- [x] Support for selecting and combining multiple strategies + doing so with cmd args.
- [x] Implement and prefix key path pruning
- [x] Make something that can use statistics with (suffix/prefix)-pruned key paths
- [x] Prune typed inner nodes (i.e., all object/array nodes)
- [ ] When max_no_paths_pruning (and min_freq ig), have some system to prioritize which paths are removed when the count is the same (like remove interior nodes before leaf nodes, or the other way around). With leaf nodes, we can still set a lower bound for the cardinality of parent nodes, though the search may be expensive. 

## STATS
- [x] Find some way to test for "key-path type confusion"
- [x] Explicitly differentiate between equi-width and singleton histograms. Update code to reflect this
- [ ] Check assumptions around sampling and what we can say about min, max, ndv and histogram stats and how we use them in estimation
- [x] Pickle and store stats of all four stat types in files. When a stat is requested, read that file instead of calculating the statistics again (if all the metadata [type, sample_rate, prune stats] matches).

## ANALYSIS
- [x] Find all key-paths, both typed and untyped
- [x] Find the value range for all key-paths
- [x] Find some strategy for picking values to test with all key-paths
  - [ ] Find an improved approach. Maybe pick edge values, median, most common, mean (if numeric), and then some random values?
- [x] Find some way to test various performance metrics for all key-paths with those test values
- [ ] Allow testing against data sets of different sizes, to compare how the techniques' performance changes with different size data sets (error should go up).
- [x] Visualize the results
- [ ] Analyze the result

## TESTING
Write tests for all important logic (wasting time on erroneous results sucks)
- [x] Estimates with basic stats
- [x] Estimates with basic_ndv stats
- [x] Estimates with hyperloglog stats
- [x] Estimates with histogram stats
- [ ] Estimates with ndv_with_mode stats
- [ ] JSON_MEMBER_OF, JSON_ARRAY_CONTAINS, JSON_ARRAY_OVERLAPS estimator functions
- [ ] Singleton histograms and Singleton_plus histograms
- [ ] Statistics creation (!!!)