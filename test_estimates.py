import math
import unittest

from compute_structures import EquiHeightBucket, Histogram, HistogramType, KeyStat, SingletonBucket, StatType
from use_stats import _update_stats_info, estimate_eq_cardinality, estimate_exists_cardinality, estimate_gt_cardinality, estimate_is_null_cardinality, estimate_lt_cardinality, estimate_not_null_cardinality, estimate_range_cardinality

# Find and replace assert with assertEquals method:
# FIND: assert ([^#]*) == ([0-9]+)
# REPLACE: self.assertEqual($1, $2)

"""
TODO: If we ever differentiate between float and int in statistics: Write tests for ints. Make num tests for floats
"""

class TestBasic(unittest.TestCase):
    def test_num_no_sampling(self):
        _update_stats_info(
            _STATS_TYPE=StatType.BASIC, 
            _stats={
                "test": KeyStat(
                    count=100,
                    null_count=20,
                ),
                "test_number": KeyStat(
                    count=80,
                    null_count=0,
                    min_val=0,
                    max_val=15
                )
            }, 
            _meta_stats={
                "highest_count_skipped": 5,
                "stats_type": StatType.BASIC,
                "sampling_rate": 0,
            }
        )


        self.assertEqual(estimate_is_null_cardinality("test"), 20)
        self.assertEqual(estimate_not_null_cardinality("test"), 80)
        self.assertEqual(estimate_exists_cardinality("test"), 100)
        self.assertEqual(estimate_exists_cardinality("test_number"), 80)
        
        # Outside value range gives 0
        self.assertEqual(estimate_eq_cardinality("test_number", -1), 0)
        self.assertEqual(estimate_eq_cardinality("test_number", 16), 0)
        # Inside value range gives EQ_MULTIPLIER * valid_count = 8 
        self.assertEqual(estimate_eq_cardinality("test_number", 5), 8)
        
        # Outside value range gives 0
        self.assertEqual(estimate_lt_cardinality("test_number", 0), 0)
        self.assertEqual(estimate_lt_cardinality("test_number", -1), 0)
        # Containing the whole value range gives all (80)
        self.assertEqual(estimate_lt_cardinality("test_number", 16), 80)
        # Containing a third of the value range [0-5) should give 80/3
        self.assertEqual(estimate_lt_cardinality("test_number", 5), 27)  # math.ceil(80/3) len(0,1,2,3,4) == 5. 5/15 = 1/3
        
        # Outside value range gives 0
        self.assertEqual(estimate_gt_cardinality("test_number", 15), 0)
        self.assertEqual(estimate_gt_cardinality("test_number", 16), 0)
        # Containing the whole value range gives all (80)
        self.assertEqual(estimate_gt_cardinality("test_number", -1), 80)
        # Containing a third of the value range (10-15] should give 80/3
        self.assertEqual(estimate_gt_cardinality("test_number", 10), 27)  # math.ceil(80/3)


        self.assertEqual(estimate_range_cardinality("test_number", range(-1, 0)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(-100000, 0)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(16, 10000)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(5, 5)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(0, 1)), 6)  # math.ceil(80/15)
        self.assertEqual(estimate_range_cardinality("test_number", range(14, 15)), 6)
        self.assertEqual(estimate_range_cardinality("test_number", range(15, 16)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(0, 5)), 27)


    def test_bool_no_sampling(self):
        # TEST BASIC WITH BOOL
        _update_stats_info(
            _STATS_TYPE=StatType.BASIC, 
            _stats={
                "test": KeyStat(
                    count=100,
                    null_count=20,
                ),
                "test_bool": KeyStat(
                    count=80,
                    min_val=False,
                    max_val=True
                )
            }, 
            _meta_stats={
                "highest_count_skipped": 5,
                "stats_type": StatType.BASIC,
                "sampling_rate": 0,
            }
        )

        self.assertEqual(estimate_exists_cardinality("test"), 100)
        self.assertEqual(estimate_is_null_cardinality("test"), 20)
        self.assertEqual(estimate_not_null_cardinality("test"), 80)
        self.assertEqual(estimate_exists_cardinality("test_bool"), 80)
        
        self.assertEqual(estimate_eq_cardinality("test_bool", True), 40)
        self.assertEqual(estimate_eq_cardinality("test_bool", False), 40)


    def test_str_no_sampling(self):
        # TEST BASIC WITH BOOL
        _update_stats_info(
            _STATS_TYPE=StatType.BASIC, 
            _stats={
                "test": KeyStat(
                    count=100,
                    null_count=20,
                ),
                "test_str": KeyStat(
                    count=80,
                    null_count=0,
                    min_val="",
                    max_val="https://www.twitter.com/zzzzzzzzzzz"
                )
            }, 
            _meta_stats={
                "highest_count_skipped": 5,
                "stats_type": StatType.BASIC,
                "sampling_rate": 0,
            }
        )

        self.assertEqual(estimate_exists_cardinality("test"), 100)
        self.assertEqual(estimate_is_null_cardinality("test"), 20)
        self.assertEqual(estimate_not_null_cardinality("test"), 80)
        self.assertEqual(estimate_exists_cardinality("test_str"), 80)
        
        self.assertEqual(estimate_eq_cardinality("test_str", ""), 8)
        self.assertEqual(estimate_eq_cardinality("test_str", "abcabc"), 8)
        self.assertEqual(estimate_eq_cardinality("test_str", "https://www.twitter.com/zzzzzzzzzzz"), 8)

    def test_arr_obj_no_sampling(self):
        _update_stats_info(
            _STATS_TYPE=StatType.BASIC, 
            _stats={
                "test": KeyStat(
                    count=100,
                    null_count=20,
                ),
                "test_obj": KeyStat(
                    count=80,
                    null_count=0,
                ),
                "test_obj.test": KeyStat(
                    count=80,
                    null_count=10
                ),
                "test_obj.test_num": KeyStat(
                    count=70,
                    null_count=0
                )
            }, 
            _meta_stats={
                "highest_count_skipped": 5,
                "stats_type": StatType.BASIC,
                "sampling_rate": 0,
            }
        )
        self.assertEqual(estimate_exists_cardinality("test"), 100)
        self.assertEqual(estimate_is_null_cardinality("test"), 20)
        self.assertEqual(estimate_not_null_cardinality("test"), 80)
        self.assertEqual(estimate_exists_cardinality("test_obj"), 80)
        self.assertEqual(estimate_exists_cardinality("test_obj.test_num"), 70)

        _update_stats_info(
            _stats={
                "test_arr": KeyStat(
                    count=100,
                    null_count=0,
                ),
                "test_arr.0_num": KeyStat(
                    count=80,
                    null_count=0
                )
            }, 
        )

        self.assertEqual(estimate_exists_cardinality("test_arr"), 100)
        self.assertEqual(estimate_is_null_cardinality("test_arr"), 0)
        self.assertEqual(estimate_not_null_cardinality("test_arr"), 100)
        


    def test_missing_no_sampling(self):
        _update_stats_info(
            _STATS_TYPE=StatType.BASIC, 
            _stats={}, 
            _meta_stats={
                "highest_count_skipped": 20,
                "stats_type": StatType.BASIC,
                "sampling_rate": 0,
            }
        )

        self.assertEqual(estimate_exists_cardinality("missingkey"), 20)
        # For next two: multiply with is_null multiplier
        self.assertEqual(estimate_is_null_cardinality("missingkey"), 2)
        self.assertEqual(estimate_not_null_cardinality("missingkey"), 18)


        # 20 * 0.1 (eq_multiplier)
        self.assertEqual(estimate_eq_cardinality("missingkey", 0), 2)
        
        # 20 * 0.3 (ineq_multiplier)
        self.assertEqual(estimate_lt_cardinality("missingkey", 0), 6)        
        self.assertEqual(estimate_gt_cardinality("missingkey", 10), 6)
        # 20 * 0.3 (range_multiplier)
        self.assertEqual(estimate_range_cardinality("missingkey", range(-1000, 10000)), 6)



    def test_missing_with_sampling(self):
        _update_stats_info(
            _STATS_TYPE=StatType.BASIC, 
            _stats={}, 
            _meta_stats={
                "highest_count_skipped": 10,
                "stats_type": StatType.BASIC,
                "sampling_rate": 0.5,
            }
        )

        self.assertEqual(estimate_exists_cardinality("missingkey"), 20)
        # For next two: multiply with is_null multiplier
        self.assertEqual(estimate_is_null_cardinality("missingkey"), 2)
        self.assertEqual(estimate_not_null_cardinality("missingkey"), 18)


        # 20 * 0.1 (eq_multiplier)
        self.assertEqual(estimate_eq_cardinality("missingkey", 0), 2)
        
        # 20 * 0.3 (ineq_multiplier)
        self.assertEqual(estimate_lt_cardinality("missingkey", 0), 6)        
        self.assertEqual(estimate_gt_cardinality("missingkey", 10), 6)
        # 20 * 0.3 (range_multiplier)
        self.assertEqual(estimate_range_cardinality("missingkey", range(-1000, 10000)), 6)


    def test_num_with_sampling(self):
        _update_stats_info(
            _STATS_TYPE=StatType.BASIC, 
            _stats={
                "test": KeyStat(
                    count=90,
                    null_count=18,
                ),
                "test_number": KeyStat(
                    count=72,
                    null_count=0,
                    min_val=0,
                    max_val=15
                )
            }, 
            _meta_stats={
                "highest_count_skipped": 4,
                "stats_type": StatType.BASIC,
                "sampling_rate": 0.1,
            }
        )

        self.assertEqual(estimate_exists_cardinality("test"), 100)
        self.assertEqual(estimate_is_null_cardinality("test"), 20)
        self.assertEqual(estimate_not_null_cardinality("test"), 80)
        
        self.assertEqual(estimate_eq_cardinality("test_number", -1), 0)
        self.assertEqual(estimate_eq_cardinality("test_number", 16), 0)
        self.assertEqual(estimate_eq_cardinality("test_number", 5), 8)
        
        self.assertEqual(estimate_lt_cardinality("test_number", 0), 0)
        self.assertEqual(estimate_lt_cardinality("test_number", -1), 0)
        self.assertEqual(estimate_lt_cardinality("test_number", 16), 80)
        self.assertEqual(estimate_lt_cardinality("test_number", 5), 27)  # math.ceil(80/3) len(0,1,2,3,4) == 5. 5/15 = 1/3
        
        self.assertEqual(estimate_gt_cardinality("test_number", 15), 0)
        self.assertEqual(estimate_gt_cardinality("test_number", 16), 0)
        self.assertEqual(estimate_gt_cardinality("test_number", -1), 80)
        self.assertEqual(estimate_gt_cardinality("test_number", 10), 27)  # math.ceil(80/3)

        self.assertEqual(estimate_range_cardinality("test_number", range(-1, 0)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(-100000, 0)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(16, 10000)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(5, 5)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(0, 1)), 6)  # math.ceil(80/15)
        self.assertEqual(estimate_range_cardinality("test_number", range(14, 15)), 6)
        self.assertEqual(estimate_range_cardinality("test_number", range(15, 16)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(0, 5)), 27)
        self.assertEqual(estimate_range_cardinality("test_number", range(-5, 5)), 27)
        self.assertEqual(estimate_range_cardinality("test_number", range(14, 100)), 6)


# TODO: Update to remove null values from typed key paths, as done in TestBasic
class TestBasicNDV(unittest.TestCase):
    STATS_TYPE = StatType.BASIC_NDV
    """
    Same tests with same values as in TestBasic, except for equality estimates,
    which should now be using the NDV statistic.
    """

    def test_num_no_sampling(self):
        _update_stats_info(
            _STATS_TYPE=self.STATS_TYPE, 
            _stats={
                "test_number": KeyStat(
                    count=100,
                    null_count=20,
                    min_val=0,
                    max_val=15,
                    ndv=8,
                )
            }, 
            _meta_stats={
                "highest_count_skipped": 5,
                "stats_type": self.STATS_TYPE,
                "sampling_rate": 0,
            }
        )


        self.assertEqual(estimate_exists_cardinality("test_number"), 100)
        self.assertEqual(estimate_is_null_cardinality("test_number"), 20)
        self.assertEqual(estimate_not_null_cardinality("test_number"), 80)
        
        # Outside value range gives 0
        self.assertEqual(estimate_eq_cardinality("test_number", -1), 0)
        self.assertEqual(estimate_eq_cardinality("test_number", 16), 0)
        # Inside value range gives (1 / ndv) * valid_count = 1/8 * 80 = 10
        self.assertEqual(estimate_eq_cardinality("test_number", 5), 10)
        
        # Outside value range gives 0
        self.assertEqual(estimate_lt_cardinality("test_number", 0), 0)
        self.assertEqual(estimate_lt_cardinality("test_number", -1), 0)
        # Containing the whole value range gives all (80)
        self.assertEqual(estimate_lt_cardinality("test_number", 16), 80)
        # Containing a third of the value range [0-5) should give 80/3
        self.assertEqual(estimate_lt_cardinality("test_number", 5), 27)  # math.ceil(80/3) len(0,1,2,3,4) == 5. 5/15 = 1/3
        
        # Outside value range gives 0
        self.assertEqual(estimate_gt_cardinality("test_number", 15), 0)
        self.assertEqual(estimate_gt_cardinality("test_number", 16), 0)
        # Containing the whole value range gives all (80)
        self.assertEqual(estimate_gt_cardinality("test_number", -1), 80)
        # Containing a third of the value range (10-15] should give 80/3
        self.assertEqual(estimate_gt_cardinality("test_number", 10), 27)  # math.ceil(80/3)


        self.assertEqual(estimate_range_cardinality("test_number", range(-1, 0)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(-100000, 0)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(16, 10000)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(5, 5)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(0, 1)), 6)  # math.ceil(80/15)
        self.assertEqual(estimate_range_cardinality("test_number", range(14, 15)), 6)
        self.assertEqual(estimate_range_cardinality("test_number", range(15, 16)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(0, 5)), 27)

    def test_bool_no_sampling(self):
        _update_stats_info(
            _STATS_TYPE=self.STATS_TYPE, 
            _stats={"test_bool": KeyStat(
                    count=100,
                    null_count=20,
                    min_val=False,
                    max_val=True,
                    ndv=2
                )
            }, 
            _meta_stats={
                "highest_count_skipped": 5,
                "stats_type": self.STATS_TYPE,
                "sampling_rate": 0,
            }
        )

        self.assertEqual(estimate_exists_cardinality("test_bool"), 100)
        self.assertEqual(estimate_is_null_cardinality("test_bool"), 20)
        self.assertEqual(estimate_not_null_cardinality("test_bool"), 80)
        
        self.assertEqual(estimate_eq_cardinality("test_bool", True), 40)
        self.assertEqual(estimate_eq_cardinality("test_bool", False), 40)


    def test_str_no_sampling(self):
        # TEST BASIC WITH BOOL
        _update_stats_info(
            _STATS_TYPE=self.STATS_TYPE, 
            _stats={"test_str": KeyStat(
                    count=100,
                    null_count=20,
                    min_val="",
                    max_val="https://www.twitter.com/zzzzzzzzzzz",
                    ndv=60,
                )
            }, 
            _meta_stats={
                "highest_count_skipped": 5,
                "stats_type": self.STATS_TYPE,
                "sampling_rate": 0,
            }
        )

        self.assertEqual(estimate_exists_cardinality("test_str"), 100)
        self.assertEqual(estimate_is_null_cardinality("test_str"), 20)
        self.assertEqual(estimate_not_null_cardinality("test_str"), 80)
        
        self.assertEqual(estimate_eq_cardinality("test_str", ""), 2)
        self.assertEqual(estimate_eq_cardinality("test_str", "abcabc"), 2)
        self.assertEqual(estimate_eq_cardinality("test_str", "https://www.twitter.com/zzzzzzzzzzz"), 2)

    def test_arr_obj_no_sampling(self):
        _update_stats_info(
            _STATS_TYPE=self.STATS_TYPE, 
            _stats={
                "test_obj": KeyStat(
                    count=100,
                    null_count=20,
                ),
                "test_obj.test_num": KeyStat(
                    count=80,
                    null_count=10
                )
            }, 
            _meta_stats={
                "highest_count_skipped": 5,
                "stats_type": self.STATS_TYPE,
                "sampling_rate": 0,
            }
        )
        self.assertEqual(estimate_exists_cardinality("test_obj"), 100)
        self.assertEqual(estimate_is_null_cardinality("test_obj"), 20)
        self.assertEqual(estimate_not_null_cardinality("test_obj"), 80)

        _update_stats_info(
            _stats={
                "test_arr": KeyStat(
                    count=100,
                    null_count=20,
                ),
                "test_arr.0_num": KeyStat(
                    count=80,
                    null_count=10
                )
            }, 
        )

        self.assertEqual(estimate_exists_cardinality("test_arr"), 100)
        self.assertEqual(estimate_is_null_cardinality("test_arr"), 20)
        self.assertEqual(estimate_not_null_cardinality("test_arr"), 80)
        


    def test_missing_no_sampling(self):
        _update_stats_info(
            _STATS_TYPE=self.STATS_TYPE, 
            _stats={}, 
            _meta_stats={
                "highest_count_skipped": 20,
                "stats_type": self.STATS_TYPE,
                "sampling_rate": 0,
            }
        )

        self.assertEqual(estimate_exists_cardinality("missingkey"), 20)
        # For next two: multiply with is_null multiplier
        self.assertEqual(estimate_is_null_cardinality("missingkey"), 2)
        self.assertEqual(estimate_not_null_cardinality("missingkey"), 18)


        # 20 * 0.1 (eq_multiplier)
        self.assertEqual(estimate_eq_cardinality("missingkey", 0), 2)
        
        # 20 * 0.3 (ineq_multiplier)
        self.assertEqual(estimate_lt_cardinality("missingkey", 0), 6)        
        self.assertEqual(estimate_gt_cardinality("missingkey", 10), 6)
        # 20 * 0.3 (range_multiplier)
        self.assertEqual(estimate_range_cardinality("missingkey", range(-1000, 10000)), 6)


    def test_missing_with_sampling(self):
        _update_stats_info(
            _STATS_TYPE=self.STATS_TYPE, 
            _stats={}, 
            _meta_stats={
                "highest_count_skipped": 10,
                "stats_type": self.STATS_TYPE,
                "sampling_rate": 0.5,
            }
        )

        self.assertEqual(estimate_exists_cardinality("missingkey"), 20)
        # For next two: multiply with is_null multiplier
        self.assertEqual(estimate_is_null_cardinality("missingkey"), 2)
        self.assertEqual(estimate_not_null_cardinality("missingkey"), 18)


        # 20 * 0.1 (eq_multiplier)
        self.assertEqual(estimate_eq_cardinality("missingkey", 0), 2)
        
        # 20 * 0.3 (ineq_multiplier)
        self.assertEqual(estimate_lt_cardinality("missingkey", 0), 6)        
        self.assertEqual(estimate_gt_cardinality("missingkey", 10), 6)
        # 20 * 0.3 (range_multiplier)
        self.assertEqual(estimate_range_cardinality("missingkey", range(-1000, 10000)), 6)


    def test_num_with_sampling(self):
        """
        Note that with this test, we assume that even though we sampled, we encountered
        not only the absolute minimum and maximum values for the key, but every 
        unique value as well. 
        This is unrealistic when large sample rates and diverse values are involved.
        """
        _update_stats_info(
            _STATS_TYPE=self.STATS_TYPE, 
            _stats={
                "test_number": KeyStat(
                    count=90,
                    null_count=18,
                    min_val=0,
                    max_val=15,
                    ndv=8
                )
            }, 
            _meta_stats={
                "highest_count_skipped": 4,
                "stats_type": self.STATS_TYPE,
                "sampling_rate": 0.1,
            }
        )

        self.assertEqual(estimate_exists_cardinality("test_number"), 100)
        self.assertEqual(estimate_is_null_cardinality("test_number"), 20)
        self.assertEqual(estimate_not_null_cardinality("test_number"), 80)
        
        self.assertEqual(estimate_eq_cardinality("test_number", -1), 0)
        self.assertEqual(estimate_eq_cardinality("test_number", 16), 0)
        self.assertEqual(estimate_eq_cardinality("test_number", 5), 10)
        
        self.assertEqual(estimate_lt_cardinality("test_number", 0), 0)
        self.assertEqual(estimate_lt_cardinality("test_number", -1), 0)
        self.assertEqual(estimate_lt_cardinality("test_number", 16), 80)
        self.assertEqual(estimate_lt_cardinality("test_number", 5), 27)  # math.ceil(80/3) len(0,1,2,3,4) == 5. 5/15 = 1/3
        
        self.assertEqual(estimate_gt_cardinality("test_number", 15), 0)
        self.assertEqual(estimate_gt_cardinality("test_number", 16), 0)
        self.assertEqual(estimate_gt_cardinality("test_number", -1), 80)
        self.assertEqual(estimate_gt_cardinality("test_number", 10), 27)  # math.ceil(80/3)

        self.assertEqual(estimate_range_cardinality("test_number", range(-1, 0)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(-100000, 0)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(16, 10000)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(5, 5)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(0, 1)), 6)  # math.ceil(80/15)
        self.assertEqual(estimate_range_cardinality("test_number", range(14, 15)), 6)
        self.assertEqual(estimate_range_cardinality("test_number", range(15, 16)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(0, 5)), 27)
        self.assertEqual(estimate_range_cardinality("test_number", range(-5, 5)), 27)
        self.assertEqual(estimate_range_cardinality("test_number", range(14, 100)), 6)


class TestHyperLogLog(TestBasicNDV):
    """
    Test that estimation functions behave in the same manner when using hyperloglog as when
    computing ndv by brute force.
    """
    STATS_TYPE = StatType.NDV_HYPERLOG

# TODO: Update to remove null values from typed key paths, as done in TestBasic
class TestHistogram(unittest.TestCase):
    def test_num_no_sampling(self):
        _update_stats_info(
            _STATS_TYPE=StatType.HISTOGRAM,
            _stats={
                "test_number": KeyStat(
                    count=120,
                    null_count=0,
                    min_val=0,
                    max_val=15,
                    histogram=Histogram(
                        HistogramType.EQUI_HEIGHT, 
                        [
                            # Upper bound, count, ndv
                            EquiHeightBucket(5, 49, 4),
                            EquiHeightBucket(9, 41, 3),
                            EquiHeightBucket(15, 30, 2)
                        ]             
                    ),
                )
            }, 
            _meta_stats={
                "stats_type": StatType.HISTOGRAM,
                "sampling_rate": 0,
            }
        )

        self.assertEqual(estimate_exists_cardinality("test_number"), 120)
        self.assertEqual(estimate_is_null_cardinality("test_number"), 0)
        self.assertEqual(estimate_not_null_cardinality("test_number"), 120)
        
        # Outside value range gives 0
        self.assertEqual(estimate_eq_cardinality("test_number", -1), 0)
        self.assertEqual(estimate_eq_cardinality("test_number", 16), 0)
        # Test inside each of the three bins 
        self.assertEqual(estimate_eq_cardinality("test_number", 5), math.ceil(49/4))
        self.assertEqual(estimate_eq_cardinality("test_number", 7), math.ceil(41/3))
        self.assertEqual(estimate_eq_cardinality("test_number", 10), math.ceil(30/2))
        
        # Outside value range gives 0
        self.assertEqual(estimate_lt_cardinality("test_number", 0), 0)
        self.assertEqual(estimate_lt_cardinality("test_number", -1), 0)
        # Containing the whole value range gives all (120)
        self.assertEqual(estimate_lt_cardinality("test_number", 16), 120)
        # Containing entirety of the first bucket
        self.assertEqual(estimate_lt_cardinality("test_number", 5), 49)
        # Containing half of the first bucket
        self.assertEqual(estimate_lt_cardinality("test_number", 2.5), math.ceil(49/2))
        # Containing first bucket and half of second bucket
        self.assertEqual(estimate_lt_cardinality("test_number", 7), 49 + math.ceil(41/2))
        
        # Outside value range gives 0
        self.assertEqual(estimate_gt_cardinality("test_number", 15), 0)
        self.assertEqual(estimate_gt_cardinality("test_number", 16), 0)
        # Containing the whole value range gives all (120)
        self.assertEqual(estimate_gt_cardinality("test_number", -1), 120)
        # Containing entirety of the last bucket
        self.assertEqual(estimate_gt_cardinality("test_number", 9), 30)
        # Containing half of the last bucket
        self.assertEqual(estimate_gt_cardinality("test_number", 12), 30//2)
        # Containing 1/6th of the last bucket
        self.assertEqual(estimate_gt_cardinality("test_number", 14), 30//6)
        # Containing last bucket and half of middle bucket
        self.assertEqual(estimate_gt_cardinality("test_number", 7), 30 + math.ceil(41/2))


        # Ranges beyond the value range give 0
        self.assertEqual(estimate_range_cardinality("test_number", range(-1, 0)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(-100000, 0)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(15, 16)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(16, 10000)), 0)
        # Zero-width range gives 0
        self.assertEqual(estimate_range_cardinality("test_number", range(5, 5)), 0)

        # Some of first bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(0, 1)), math.ceil(49/5))
        # Some of last bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(14, 15)), math.ceil(30/6))
        # All of first bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(0, 5)), 49)
        # All of middle bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(5, 9)), 41)
        # 2/5th of first bucket plus all of middle bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(3, 9)), math.ceil(49*(2/5)) + 41)
        # Half of middle bucket plus all of last bucket, but with range extending beyond
        self.assertEqual(estimate_range_cardinality("test_number", range(7, 100)), math.ceil(41/2) + 30)
        # From far below, all of first bucket plus 1/4 of middle bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(-1000, 6)), 49 + math.ceil(41/4))

    def test_num_singleton_no_sampling(self):
        _update_stats_info(
            _STATS_TYPE=StatType.HISTOGRAM,
            _stats={
                "test_number": KeyStat(
                    count=140,
                    null_count=60,
                    min_val=-0.5,
                    max_val=10000,
                    histogram=Histogram(
                        HistogramType.EQUI_HEIGHT, 
                        [
                            # Upper bound, count, ndv
                            SingletonBucket(-0.5, 40),
                            SingletonBucket(5.5, 20),
                            SingletonBucket(10, 40),
                            SingletonBucket(10000, 40),
                        ]
                    ),
                )
            }, 
            _meta_stats={
                "stats_type": StatType.HISTOGRAM,
                "sampling_rate": 0,
            }
        )

        return 
        # TODO!!

        self.assertEqual(estimate_exists_cardinality("test_number"), 200)
        self.assertEqual(estimate_is_null_cardinality("test_number"), 60)
        self.assertEqual(estimate_not_null_cardinality("test_number"), 140)
        
        # Outside value range gives 0
        self.assertEqual(estimate_eq_cardinality("test_number", -0.6), 0)
        self.assertEqual(estimate_eq_cardinality("test_number", 10_001), 0)
        # Test between bins 
        self.assertEqual(estimate_eq_cardinality("test_number", 5), math.ceil(49/4))
        self.assertEqual(estimate_eq_cardinality("test_number", 7), math.ceil(41/3))
        self.assertEqual(estimate_eq_cardinality("test_number", 10), math.ceil(30/2))
        # Test right on bins
        
        # Outside value range gives 0
        self.assertEqual(estimate_lt_cardinality("test_number", 0), 0)
        self.assertEqual(estimate_lt_cardinality("test_number", -1), 0)
        # Containing the whole value range gives all (120)
        self.assertEqual(estimate_lt_cardinality("test_number", 16), 120)
        # Containing entirety of the first bucket
        self.assertEqual(estimate_lt_cardinality("test_number", 5), 49)
        # Containing half of the first bucket
        self.assertEqual(estimate_lt_cardinality("test_number", 2.5), math.ceil(49/2))
        # Containing first bucket and half of second bucket
        self.assertEqual(estimate_lt_cardinality("test_number", 7), 49 + math.ceil(41/2))
        
        # Outside value range gives 0
        self.assertEqual(estimate_gt_cardinality("test_number", 15), 0)
        self.assertEqual(estimate_gt_cardinality("test_number", 16), 0)
        # Containing the whole value range gives all (120)
        self.assertEqual(estimate_gt_cardinality("test_number", -1), 120)
        # Containing entirety of the last bucket
        self.assertEqual(estimate_gt_cardinality("test_number", 9), 30)
        # Containing half of the last bucket
        self.assertEqual(estimate_gt_cardinality("test_number", 12), 30//2)
        # Containing 1/6th of the last bucket
        self.assertEqual(estimate_gt_cardinality("test_number", 14), 30//6)
        # Containing last bucket and half of middle bucket
        self.assertEqual(estimate_gt_cardinality("test_number", 7), 30 + math.ceil(41/2))


        # Ranges beyond the value range give 0
        self.assertEqual(estimate_range_cardinality("test_number", range(-1, 0)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(-100000, 0)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(15, 16)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(16, 10000)), 0)
        # Zero-width range gives 0
        self.assertEqual(estimate_range_cardinality("test_number", range(5, 5)), 0)

        # Some of first bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(0, 1)), math.ceil(49/5))
        # Some of last bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(14, 15)), math.ceil(30/6))
        # All of first bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(0, 5)), 49)
        # All of middle bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(5, 9)), 41)
        # 2/5th of first bucket plus all of middle bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(3, 9)), math.ceil(49*(2/5)) + 41)
        # Half of middle bucket plus all of last bucket, but with range extending beyond
        self.assertEqual(estimate_range_cardinality("test_number", range(7, 100)), math.ceil(41/2) + 30)
        # From far below, all of first bucket plus 1/4 of middle bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(-1000, 6)), 49 + math.ceil(41/4))

    def test_bool_no_sampling(self):
        _update_stats_info(
            _STATS_TYPE=StatType.HISTOGRAM, 
            _stats={"test_bool": KeyStat(
                    count=100,
                    null_count=20,
                    min_val=False,
                    max_val=True,
                    histogram=Histogram(
                        HistogramType.EQUI_HEIGHT,
                        [EquiHeightBucket(False, 5, 1), EquiHeightBucket(True, 75, 1)]
                    ),
                )
            },
            _meta_stats={
                "highest_count_skipped": 5,
                "stats_type": StatType.HISTOGRAM,
                "sampling_rate": 0,
            }
        )

        self.assertEqual(estimate_exists_cardinality("test_bool"), 100)
        self.assertEqual(estimate_is_null_cardinality("test_bool"), 20)
        self.assertEqual(estimate_not_null_cardinality("test_bool"), 80)
        
        self.assertEqual(estimate_eq_cardinality("test_bool", True), 75)
        self.assertEqual(estimate_eq_cardinality("test_bool", False), 5)


    def test_str_no_sampling(self):
        _update_stats_info(
            _STATS_TYPE=StatType.HISTOGRAM, 
            _stats={"test_str": KeyStat(
                    count=100,
                    null_count=5,
                    min_val="",
                    max_val="zzzzz",
                    histogram=Histogram(
                        HistogramType.SINGLETON,
                        [
                            SingletonBucket("", 5), SingletonBucket("abc", 75), 
                            SingletonBucket("abcc", 1), SingletonBucket("zabc", 2), 
                            SingletonBucket("zz", 2), SingletonBucket("zzzzz", 10)
                        ]
                    ),
                )
            },
            _meta_stats={
                "stats_type": StatType.HISTOGRAM,
                "sampling_rate": 0,
            }
        )

        self.assertEqual(estimate_exists_cardinality("test_str"), 100)
        self.assertEqual(estimate_is_null_cardinality("test_str"), 5)
        self.assertEqual(estimate_not_null_cardinality("test_str"), 95)
        
        # Test strings not in the dataset
        self.assertEqual(estimate_eq_cardinality("test_str", "missingstring"), 0)
        self.assertEqual(estimate_eq_cardinality("test_str", "ab"), 0)
        self.assertEqual(estimate_eq_cardinality("test_str", "z"), 0)

        # Test strings present in the data
        self.assertEqual(estimate_eq_cardinality("test_str", ""), 5)
        self.assertEqual(estimate_eq_cardinality("test_str", "abc"), 75)
        self.assertEqual(estimate_eq_cardinality("test_str", "abcc"), 1)
        self.assertEqual(estimate_eq_cardinality("test_str", "zzzzz"), 10)


    def test_missing_no_sampling(self):
        _update_stats_info(
            _STATS_TYPE=StatType.HISTOGRAM, 
            _stats={}, 
            _meta_stats={
                "highest_count_skipped": 10,
                "stats_type": StatType.HISTOGRAM,
                "sampling_rate": 0.5,
            }
        )

        self.assertEqual(estimate_exists_cardinality("missingkey"), 20)
        # For next two: multiply with is_null multiplier
        self.assertEqual(estimate_is_null_cardinality("missingkey"), 2)
        self.assertEqual(estimate_not_null_cardinality("missingkey"), 18)


        # 20 * 0.1 (eq_multiplier)
        self.assertEqual(estimate_eq_cardinality("missingkey", 0), 2)
        
        # 20 * 0.3 (ineq_multiplier)
        self.assertEqual(estimate_lt_cardinality("missingkey", 0), 6)        
        self.assertEqual(estimate_gt_cardinality("missingkey", 10), 6)
        # 20 * 0.3 (range_multiplier)
        self.assertEqual(estimate_range_cardinality("missingkey", range(-1000, 10000)), 6)

    def test_missing_with_sampling(self):
        _update_stats_info(
            _STATS_TYPE=StatType.HISTOGRAM, 
            _stats={}, 
            _meta_stats={
                "highest_count_skipped": 10,
                "stats_type": StatType.HISTOGRAM,
                "sampling_rate": 0.5,
            }
        )

        self.assertEqual(estimate_exists_cardinality("missingkey"), 20)
        # For next two: multiply with is_null multiplier
        self.assertEqual(estimate_is_null_cardinality("missingkey"), 2)
        self.assertEqual(estimate_not_null_cardinality("missingkey"), 18)


        # 20 * 0.1 (eq_multiplier)
        self.assertEqual(estimate_eq_cardinality("missingkey", 0), 2)
        
        # 20 * 0.3 (ineq_multiplier)
        self.assertEqual(estimate_lt_cardinality("missingkey", 0), 6)        
        self.assertEqual(estimate_gt_cardinality("missingkey", 10), 6)
        # 20 * 0.3 (range_multiplier)
        self.assertEqual(estimate_range_cardinality("missingkey", range(-1000, 10000)), 6)


    def test_num_with_sampling(self):
        # Again. Assumptions made here are completely unrealistic and should be revisited
        _update_stats_info(
            _STATS_TYPE=StatType.HISTOGRAM,
            _stats={
                "test_number": KeyStat(
                    count=120,
                    null_count=0,
                    min_val=0,
                    max_val=15,
                    histogram=Histogram(
                        HistogramType.EQUI_HEIGHT,
                        [
                            # Upper bound, count, ndv
                            EquiHeightBucket(5, 49, 4),
                            EquiHeightBucket(9, 41, 3),
                            EquiHeightBucket(15, 30, 2)
                        ]                
                    ),
                )
            }, 
            _meta_stats={
                "stats_type": StatType.HISTOGRAM,
                "sampling_rate": 0.2,
            }
        )

        self.assertEqual(estimate_exists_cardinality("test_number"), 150)
        self.assertEqual(estimate_is_null_cardinality("test_number"), 0)
        self.assertEqual(estimate_not_null_cardinality("test_number"), 150)
        
        # Outside value range gives 0
        self.assertEqual(estimate_eq_cardinality("test_number", -1), 0)
        self.assertEqual(estimate_eq_cardinality("test_number", 16), 0)
        # Test inside each of the three bins 
        self.assertEqual(estimate_eq_cardinality("test_number", 5), math.ceil(1.25 * 49/4))
        self.assertEqual(estimate_eq_cardinality("test_number", 7), math.ceil(1.25 * 41/3))
        self.assertEqual(estimate_eq_cardinality("test_number", 10), math.ceil(1.25 * 30/2))
        
        # Outside value range gives 0
        self.assertEqual(estimate_lt_cardinality("test_number", 0), 0)
        self.assertEqual(estimate_lt_cardinality("test_number", -1), 0)
        # Containing the whole value range gives all (120)
        self.assertEqual(estimate_lt_cardinality("test_number", 16), 150)
        # Containing entirety of the first bucket
        self.assertEqual(estimate_lt_cardinality("test_number", 5), math.ceil(1.25*49))
        # Containing half of the first bucket
        self.assertEqual(estimate_lt_cardinality("test_number", 2.5), math.ceil(1.25 * 49/2))
        # Containing first bucket and half of second bucket
        self.assertEqual(estimate_lt_cardinality("test_number", 7), math.ceil(1.25*(49 + 41/2)))
        
        # Outside value range gives 0
        self.assertEqual(estimate_gt_cardinality("test_number", 15), 0)
        self.assertEqual(estimate_gt_cardinality("test_number", 16), 0)
        # Containing the whole value range gives all (120)
        self.assertEqual(estimate_gt_cardinality("test_number", -1), 150)
        # Containing entirety of the last bucket
        self.assertEqual(estimate_gt_cardinality("test_number", 9), 38)
        # Containing half of the last bucket
        self.assertEqual(estimate_gt_cardinality("test_number", 12), 19)
        # Containing 1/6th of the last bucket
        self.assertEqual(estimate_gt_cardinality("test_number", 14), 7)
        # Containing last bucket and half of middle bucket
        self.assertEqual(estimate_gt_cardinality("test_number", 7), math.ceil(1.25 * (30 + 41/2)))


        # Ranges beyond the value range give 0
        self.assertEqual(estimate_range_cardinality("test_number", range(-1, 0)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(-100000, 0)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(15, 16)), 0)
        self.assertEqual(estimate_range_cardinality("test_number", range(16, 10000)), 0)
        # Zero-width range gives 0
        self.assertEqual(estimate_range_cardinality("test_number", range(5, 5)), 0)

        # Some of first bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(0, 1)), math.ceil(1.25 * 49/5))
        # Some of last bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(14, 15)), math.ceil(1.25 * 30/6))
        # All of first bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(0, 5)), math.ceil(1.25 * 49))
        # All of middle bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(5, 9)), math.ceil(1.25 * 41))
        # 2/5th of first bucket plus all of middle bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(3, 9)), math.ceil(1.25 *(49*(2/5) + 41)))
        # Half of middle bucket plus all of last bucket, but with range extending beyond
        self.assertEqual(estimate_range_cardinality("test_number", range(7, 100)), math.ceil(1.25 *(41/2 + 30)))
        # From far below, all of first bucket plus 1/4 of middle bucket
        self.assertEqual(estimate_range_cardinality("test_number", range(-1000, 6)), math.ceil(1.25 * (49 +  41/4)))