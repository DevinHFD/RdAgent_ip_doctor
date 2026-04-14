| Feature Name | Description | Type |
|-------------|------------|------|
| WAC | Gross coupon paid at origination | float |
| Purchase_Pct | % of UPB from new purchase | float |
| Refinance_Pct | % of UPB from refinancing | float |
| Cashout_Pct | % of UPB Cashout Refi Loans | float |
| Twofour_Unit_Pct | % of UPB 2-4 unit homes | float |
| Orig_Avg_Loan_Size | Original average loan size | float |
| Orig_LTV | Loan-to-Value ratio at origination | float |
| Orig_FICO | Credit score at origination | float |
| Channel_Broker | % of UPB from broker channel | float |
| Channel_Correspondent | % of UPB from correspondent channel | float |
| Loan_Size_Dispersion | (Loan_Size_75 - Loan_Size_25) / Orig_Avg_Loan_Size | float |
| FICO_Dispersion | (FICO_75 - FICO_25) / Orig_FICO | float |
| LTV_Dispersion | (Orig_LTV_75 - Orig_LTV_25) / Orig_LTV | float |
| Servicer_Wellsfargo | % of UPB serviced by Wells Fargo | float |
| Servicer_Quicken | % of UPB serviced by Quicken | float |
| Servicer_Unitedwholesales | % of UPB serviced by United Wholesale | float |
| Servicer_Pennymac | % of UPB serviced by Penny Mac | float |
| Servicer_Freedommtge | % of UPB serviced by Freedom Mortgage | float |
| Other_Servicer_Bank | % of UPB serviced by Other Bank Servicers | float |
| Other_Servicer_NonBank | % of UPB serviced by Other Non Bank Servicers | float |
| CA_pctRPB | % of UPB in CA | float |
| NY_pctRPB | % of UPB in NY | float |
| FL_pctRPB | % of UPB in FL | float |
| TX_pctRPB | % of UPB in TX | float |
| PR_pctRPB | % of UPB in PR | float |
| Pacific_pctRPB | Geographic area control: Pacific states | float |
| Mountain_pctRPB | Geographic area control: Mountain states | float |
| Eastsouthcentral_pctRPB | Geographic area control: East South Central States | float |
| Westsouthcentral_pctRPB | Geographic area control: West South Central States | float |
| Midwest_pctRPB | Geographic area control: Midwestern States | float |
| Southatlantic_pctRPB | Geographic area control: South Atlantic States | float |
| Midatlantic_pctRPB | Geographic area control: Mid-Atlantic States | float |
| Newengland_pctRPB | Geographic area control: New England States | float |
| Orig_LTV_Missing | Indicator: Orig_LTV is not available in raw data | float |
| Orig_FICO_Missing | Indicator: Orig_FICO is not populated in raw data | float |
| FHA_Program | % of UPB from Federal Housing Administration | float |
| VA_Program | % of UPB from Department of Veterans Affairs | float |
| SATO | Spread between WAC and Mortgage Rate at Origination | float |
| Avg_Prop_Refi_Incentive_WAC_30yr_2mos | Refinance Incentive: 30yr | float |
| Avg_Prop_Switch_To_15yr_Incentive_2mos | Refinance Incentive: 30yr to 15yr | float |
| Burnout_Prop_WAC_30yr_log_sum60 | Burnout: 30yr, log sum over last 60 months | float |
| Burnout_Prop_30yr_Switch_to_15_Lag1 | Burnout: 30yr to 15yr, lagged by 1 month | float |
| CLTV | Current LTV (CLTV) | float |
| Coll_HPA_2yr | Pool level HPA over the last 2 yrs | float |
| WALA_less_eq_6 | Indicator for WALA ≤ 6 | int |
| WALA_bet_7_and_12 | Indicator for WALA between 7 and 12 | int |
| WALA | Weighted Average Loan Age | float |
| Num_Business_Days | Number of Business Days in Month | int |
| Perf_after2008 | Indicator: Performance month after 2008 | int |
| is_Jan | Indicator: January | int |
| is_Feb | Indicator: February | int |
| is_Mar | Indicator: March | int |
| is_Apr | Indicator: April | int |
| is_May | Indicator: May | int |
| is_Jun | Indicator: June | int |
| is_Jul | Indicator: July | int |
| is_Aug | Indicator: August | int |
| is_Sep | Indicator: September | int |
| is_Oct | Indicator: October | int |
| is_Nov | Indicator: November | int |