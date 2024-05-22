import polars as pl


def date_cols() -> dict:
    """
    Dict for aggregating rows by max date for tables with depth 1 and 2
    Key - table name,
    Value - max_date column

    Returns
    -------
    out: dict
    """
    return {
        'applprev_1_*': 'creationdate_885D',
        'tax_registry_a_1': 'recorddate_4527225D',
        'tax_registry_b_1': 'deductiondate_4917603D',
        'tax_registry_c_1': 'processingdate_168D',
        'credit_bureau_a_1_*': 'refreshdate_3813885D',
        'credit_bureau_b_1': 'lastupdate_260D',
        'person_1': 'empl_employedfrom_271D',
        'credit_bureau_b_2': 'pmts_date_1107D',
        'person_2': 'empls_employedfrom_796D',
        'applprev_2': None,
        'credit_bureau_a_2_*': None
    }


def ignored_features() -> dict:
    """
    Dict with features to remove.
    Key - table name,
    Value - list with exact name of column or substring

    Returns
    -------
    out: dict
    """
    return {
        'static_cb_0': [
            'contractssum',
            'description',
            'for',
            'assignmentdate',
            'pmtaverage',
            'pmtcount',
            'riskassesment',
            'maritalst_893M',
        ],
        'static_0': [
            'avglnamtstart24m', 'avgpmtlast12m', 'bankacctype', 'cardtype',
            'datelastinstal', 'equality', 'inittransaction',
            'lastdependentsnum', 'lastother', 'lastrepayingdate',
            'mastercontre', 'maxannuity_4075009A', 'maxlnamt', 'maxpmtlast',
            'payvacationpostpone', 'paytype', 'totinstallast1m', 'typesuite',
            'validfrom', 'commnoinclast6m', 'deferredmnthsnum',
#             'pctinstlsallpaidlate4d', 'pctinstlsallpaidlate6d',
#             'cntincpaycont9m', 'monthsannuity', 'maxdebt4_972A',
#             'maxinstallast24m',
#             'numincomingpmts_3546848L', 'numinstls_657L',
#             'numinstregularpaid', 'numinstpaidlastcontr_4325080L',
#             'numinstpaid_4499208L', 'numinstlsallpaid_934L', 
#             'sellerplace',
#             'maxoutstandbalancel12m',
#             'actualdpd', 'opencred_647L'
        ],
        'applprev_1_*': [
            'byoccupationinc', 'credacc_maxhisbal', 'credacc_minhisbal',
            'credacc_status', 'credacc_transactions', 'revolvingaccount',
#             'credacc_actualbalance', 'mainoccupationinc', 'cancelreason',
#             'credacc_credlmt', 'actualdpd', 'downpmt'
        ],
        'credit_bureau_a_1_*': [
            'contractsum', 'description', 'dpdmaxdatemonth', 'dpdmaxdateyear',
            'overdueamountmaxdatemonth', 'overdueamountmaxdateyear',
            'prolongationcount', 'refreshdate', 'instlamount_852A',
#             'monthlyinstlamount',
#             'numberofoverdueinstls_834L',
#             'dateofrealrepmt',
            'lastupdate_388D'
        ],
        'credit_bureau_b_1': [
            'dpd_',
            'dpdmaxdatemonth',
            'dpdmaxdateyear_742T',
            'lastupdate',
            'maxdebtpduevalodued',
            'overdueamountmaxdatemonth',
            'overdueamountmaxdateyear',
            'dpdmax_',
            'contractdate',
            'contracttype',
            'credor',
            'debtpastduevalue',
            'installmentamount_644A',
            'overdueamountmax_950A',
            'periodicityofpmts_997',
            'pmtdaysoverdue',
            'pmtmethod',
        ],
        'person_1': [
            'hous', 'isreference', 'role_993L', 'empl_employedtotal',
            'maritalst',
#             'role_1084L', 
#             'type_25L',
#             'empladdr_district_926M',
#             'empladdr_zipcode_114M',
#             'mainoccupationinc', 
#             'num_group'
        ],
        'person_2': [
            'addres', 'conts_role', 'empls_economicalst',
            'relatedpersons_role', 'empls_employedfrom'
        ],
        'credit_bureau_a_2_*': [
            'collater_typofvalofguarant', 'collater_valueofguarantee',
            'pmts_month', 'pmts_year'
        ],
        'applprev_2': ['conts_type', 'credacc_cards_status'],
        'main': ['num_group1_cb_b',
#                  'lastcancelreason_561M'
                 'totalsettled_863A','mainoccupationinc_437A', 'amtinstpaidbefduel24m_4187115A'
                ]
    }


def education_grades() -> dict:
    """
    Dict with education column label encoder
    Key - old value,
    Value - new value.

    Returns
    -------
    out: dict
    """
    return {
        'None': 1,
        'a55475b1': 1,
        '6b2ae0fa': 2,
        'P97_36_170': 2,
        '717ddd49': 3,
        'P33_146_175': 3,
        '39a0853f': 4,
        'a34a13c8': 4,
        'P106_81_188': 4,
        'c8e1a1d0': 5,
        'P17_36_170': 5,
        'P157_18_172': 6
    }


# region AGGREGATION PARAMETERS

def agg_col_params() -> dict[str, dict]:
    """
    Dict of dicts for columns aggregation.

    Returns
    -------
    out: dict(dict)
    """
    return {
        'static_cb': agg_col_static_cb(),
        'static_0': agg_col_static_0(),
        'credit_bureau_a_1_*': agg_col_cb_a_1(),
        'credit_bureau_b_1': agg_col_cb_b_1(),
        'person_1': agg_col_person_1(),
        'applprev_1_*': agg_col_applprev_1(),
        'main': agg_col_main()
    }


def agg_col_static_cb() -> dict:
    """
    Dict for columns aggregation in static_cb table
    Key - new alias
    Value - tuple
        0 - substring for feature search /
            list of feature names /
            dict with key as substring for search and values as additional features
        1 - dtype
        2 - True to use min_horizontal, False (default) use max_horizontal, optional.
    Returns
    -------
    out: dict
    """
    return {
        'birthdate_0': ('birth', pl.Date),
        'number_results_year': ('quarter', pl.Float64),
        'responsedate': ('responsedate', pl.Date),
    }


def agg_col_static_0() -> dict:
    """
    Dict for columns aggregation in static_0 table
    Key - new alias
    Value - tuple
        0 - substring for feature search /
            list of feature names /
            dict with key as substring for search and values as additional features
        1 - dtype
        2 - True to use min_horizontal, False (default) use max_horizontal, optional.
    Returns
    -------
    out: dict
    """
    return {
        'applicationcnt_for_client': ([
                                          'applicationcnt_361L', 'applications30d_658L',
                                          'applicationscnt_1086L', 'applicationscnt_867L'
                                      ], pl.Int32),
        'applicationcnt_for_empl':
            (['applicationscnt_464L', 'applicationscnt_629L'], pl.Int32),
        'avg_days_past_or_before':
            (['avgdbddpdlast24m_3658932P', 'avgdbddpdlast3m_4187120P'], pl.Int32),
        'avg_days_past':
            (['avgdpdtolclosure24_3658938P',
              'avgmaxdpdlast9m_3716943P'], pl.Int32),
        'sumoutstandtotal': ('sumoutstandtotal', pl.Int32),
        'num_active_creds': ('numactive', pl.Int32),
        'maxdbddpd': ('maxdbddpd', pl.Int32),
        'isbidproduct': ('isbidproduct', pl.Boolean),
        'currdebt': ('currdebt', pl.Float64),
        'clients_cnt': ('clientscnt_', pl.Int32),
        'clients_cnt_same_phone': ('clientscnt', pl.Int32),
        'date_first_offer_campaign':
            (['datefirstoffer_1144D', 'firstclxcampaign_1125D'], pl.Date),
        'max_num_unpaid_inst': ({
                                    'numinsttopaygr': None,
                                    'numinstunpaidmax': None
                                }, pl.Int32),
        'interestrate': ({
                             'interestrate': 'eir_270L'
                         }, pl.Float64),
        'max_dpd': ({
                        'maxdpdlast': None,
                        'maxdpdtolerance': 'maxdpdfrom6mto36m_3546853P'
                    }, pl.Int32),
        'num_early_paid_inst': ({
                                    'numinstpaidearly': [
                                        'numinstlallpaidearly3d_817L', 'numinstlswithoutdpd_562L',
                                        'numinstmatpaidtearly2d_4499204L'
                                    ]
                                }, pl.Int32),
        'num_overdue_paid_inst': ({
                                      'numinstlswithdpd': None,
                                      'numinstpaidlate1d': None
                                  }, pl.Int32),
    }


def agg_col_cb_a_1() -> dict:
    """
    Dict for columns aggregation in credit_bureau_a_1 tables
    Key - new alias
    Value - tuple
        0 - substring for feature search /
            list of feature names /
            dict with key as substring for search and values as additional features
        1 - dtype
        2 - True to use min_horizontal, False (default) use max_horizontal, optional.
    Returns
    -------
    out: dict
    """
    return {
        'interestrate': ({
                             'annualeffectiverate': 'interestrate_508L',
                             'nominalrate': None
                         }, pl.Float64),
        'max_overdueamount_closed': ([
                                         'overdueamount_31A', 'overdueamountmax_35A',
                                         'overdueamountmax2_398A'
                                     ], pl.Float64),
        'max_overdueamount_active': ([
                                         'overdueamount_659A', 'overdueamountmax_155A',
                                         'overdueamountmax2_14A'
                                     ], pl.Float64)
    }


def agg_col_cb_b_1() -> dict:
    """
    Dict for columns aggregation in credit_bureau_b_1 tables
    Key - new alias
    Value - tuple
        0 - substring for feature search /
            list of feature names /
            dict with key as substring for search and values as additional features
        1 - dtype
        2 - True to use min_horizontal, False (default) use max_horizontal, optional.
    Returns
    -------
    out: dict
    """
    return {
        'credlmt_active': (['credlmt_1052A', 'credlmt_3940954A'], pl.Float64),
        'interestrate': ('interest', pl.Float64),
        'residualamount_active':
            (['residualamount_127A', 'residualamount_3940956A'], pl.Float64)
    }


def agg_col_person_1() -> dict:
    """
    Dict for columns aggregation in person_1 table
    Key - new alias
    Value - tuple
        0 - substring for feature search /
            list of feature names /
            dict with key as substring for search and values as additional features
        1 - dtype
        2 - True to use min_horizontal, False (default) use max_horizontal, optional.
    Returns
    -------
    out: dict
    """
    return {
        'birthdate': ('birth', pl.Date),
        'contaddr_zipcode': ([
                                 'contaddr_district_15M', 'contaddr_zipcode_807M',
                                 'registaddr_district_1083M', 'registaddr_zipcode_184M'
                             ], pl.String),
        'contaddr_smempladdr':
            (['contaddr_matchlist_1032L', 'contaddr_smempladdr_334L'], pl.Boolean),
        'persontype': ('persontype', pl.Int32),
        'relationshiptoclient': ('relationshiptoclient', pl.String),
        'gender': ({
                       'gender': None,
                       'sex': None
                   }, pl.String)
    }


def agg_col_main() -> dict:
    """
    Dict for columns aggregation in main (base table + left join of rest) table
    Key - new alias
    Value - tuple
        0 - substring for feature search /
            list of feature names /
            dict with key as substring for search and values as additional features
        1 - dtype
        2 - True to use min_horizontal, False (default) use max_horizontal, optional.
    Returns
    -------
    out: dict
    """
    return {
        'total_outstand_amount_closed_credit':
            (['totaloutstanddebtvalue_668A',
              'outstandingamount_354A'], pl.Float64),
        'numberofqueries_year':
            (['number_results_year', 'numberofqueries_373L',
              'days360_512L'], pl.Int32),
        'days_90_120_180': (['days90_310L', 'days120_123L',
                             'days180_256L'], pl.Int32),
        'tax_amount_max': (['pmtssum_45A', 'tax_amount'], pl.Float64),
        'record_response_days': (['record_date', 'responsedate'], pl.Int32),
        'last_unpaid_delinq_days_ago':
            (['datelastunpaid_3546854D', 'lastdelinqdate_224D'], pl.Int32),
        'max_dpd_overdue': ([
                                'daysoverduetolerancedd_3976961L', 'max_dpd',
                                'maxdpdtolerance_577P'
                            ], pl.Int32),
        'last_payment_days_ago': ('dtlastpmt', pl.Int32),
        'last_activ_appr_days_ago': ([
                                         'lastactivateddate_801D', 'lastapprdate_640D', 'approvaldate_319D',
                                         'dateactivated_425D'
                                     ], pl.Int32),
        'prev_appl_days_ago': ([
                                   'lastapplicationdate_877D', 'creationdate_885D',
                                   'firstnonzeroinstldate_307D'
                               ], pl.Int32),
        'min_days_before_or_past_due': ('mindbd', pl.Int32),
        'debt_outstand_total': ([
                                    'totaldebt_9A', 'sumoutstandtotal', 'currdebt', 'currdebt_94A',
                                    'outstandingdebt_522A'
                                ], pl.Float64),
        'childnum': ('childnum', pl.Int32),
        'employedfrom': (['employedfrom_700D',
                          'empl_employedfrom_271D'], pl.Int32),
        'num_group1_applprev_2_min': (['num_group1', 'num_group1_applprev_2'],
                                      pl.Int32, True),
        'pmt_num_prev_appl': (['pmtnum_8L', 'tenor_203L'], pl.Int32),
        'credlmt_closed': (['credlmt_228A', 'credlmt_230A'], pl.Float64),
        'credlmt_actv': (['credlmt_935A', 'credlmt_active'], pl.Float64),
        'active_contract_end_date_days_ago':
            (['contractmaturitydate_151D', 'dateofcredend_289D'], pl.Int32, True),
        'outstand_amount_debt_active': ([
                                            'debtoutstand_525A', 'outstandingamount_362A',
                                            'totaloutstanddebtvalue_39A', 'debtvalue_227A'
                                        ], pl.Float64),
        'debt_overdue': (['debtoverdue_47A',
                          'totaldebtoverduevalue_178A'], pl.Float64),
        'num_cred_active': (['credquantity_1099L',
                             'numberofcontrsvalue_258L'], pl.Int32),
        'num_cred_closed': (['credquantity_984L',
                             'numberofcontrsvalue_358L'], pl.Int32),
        'num_instls_active': (['numberofinstls_320L',
                               'numberofinstls_810L'], pl.Int32),
        'num_outstand_pending_instls_active':
            (['numberofoutstandinstls_59L', 'pmtnumpending_403L'], pl.Int32),
        'residual_amount_active':
            (['residualamount_856A', 'residualamount_active'], pl.Float64),
        'total_amount_closed_contracts':
            (['totalamount_6A', 'totalamount_881A'], pl.Float64),
        'total_amount_active_contracts':
            (['totalamount_996A', 'totalamount_503A', 'amount_1115A'], pl.Float64),
        'interestrate_cb': ('interestrate_cb', pl.Float64),
        'education_max': ('education', pl.Int32),
        'avg_days_before_or_past_due':
            (['avg_days_past_or_before', 'avgdbdtollast24m_4525197P'], pl.Int32),
        'classificationofcontr_active':
            (['classificationofcontr_1114M',
              'classificationofcontr_13M'], pl.String),
        'contractst_active': (['contractst_516M',
                               'contractst_545M'], pl.String),
        'employers_name': (['employer_name',
                            'empls_employer_name_740M'], pl.String),
        'familystate': ('familystate', pl.String),
        'installmentamount_active':
            (['installmentamount_833A', 'instlamount_892A',
              'instlamount_768A'], pl.Float64),
        'isdebitcard': ('isdebitcard', pl.Int8),
        'purposeofcred_active': (['purposeofcred_722M',
                                  'purposeofcred_426M'], pl.String),
        'residualamount_closed':
            (['residualamount_1093A', 'residualamount_488A'], pl.Float64),
        'subjectrole_active': (['subjectrole_182M',
                                'subjectrole_326M'], pl.String),
        'subjectrole_closed': (['subjectrole_43M',
                                'subjectrole_93M'], pl.String),
        'empladdr': (['empladdr_district_926M', 'empladdr_zipcode_114M'], pl.String),
        'num_total_inst': (['numincomingpmts_3546848L', 'numinstls_657L'], pl.Int32),
        'num_total_paid_inst': ({'numinstregularpaid':['numinstpaidlastcontr_4325080L', 
                                 'numinstpaid_4499208L', 'numinstlsallpaid_934L']}, pl.Int32),
        'cred_closure_date_days_ago': (['dateofcredend_353D', 'dateofrealrepmt_138D'], pl.Int32, True),
        'actualdpd': ('actualdpd', pl.Int32),
        'sellerplace_cnt': ('sellerplace', pl.Int32)
    }


def agg_col_applprev_1() -> dict:
    """
    Dict for columns aggregation in applprev_1 table
    Key - new alias
    Value - tuple
        0 - substring for feature search /
            list of feature names /
            dict with key as substring for search and values as additional features
        1 - dtype
        2 - True to use min_horizontal, False (default) use max_horizontal, optional.
    Returns
    -------
    out: dict
    """
    return {'dtlastpmt': ('dtlastpmt', pl.Date)}

# endregion
