import pandas as pd
import scipy.stats as stats

class InsuranceDataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

    def test_risk_differences_across_provinces(self):
        # Group data by Province and compute TotalClaims
        province_groups = self.data.groupby('Province')['TotalClaims'].mean()
        _, p_value = stats.f_oneway(*[self.data[self.data['Province'] == province]['TotalClaims']
                                     for province in province_groups.index])

        # Interpret the result
        alpha = 0.05
        if p_value < alpha:
            print("Reject the null hypothesis: There are risk differences across provinces.")
        else:
            print("Fail to reject the null hypothesis: No significant risk differences across provinces.")
    
    def test_risk_differences_between_zip_codes(self):
        # Group data by PostalCode and compute TotalClaims
        zip_code_groups = self.data.groupby('PostalCode')['TotalClaims'].mean()
        _, p_value = stats.f_oneway(*[self.data[self.data['PostalCode'] == zip_code]['TotalClaims']
                                     for zip_code in zip_code_groups.index])

        # Interpret the result
        alpha = 0.05
        if p_value < alpha:
            print("Reject the null hypothesis: There are risk differences between zip codes.")
        else:
            print("Fail to reject the null hypothesis: No significant risk differences between zip codes.")
    
    def test_margin_differences_between_zip_codes(self):
        # Group data by PostalCode and compute TotalPremium
        zip_code_groups = self.data.groupby('PostalCode')['TotalPremium'].mean()
        _, p_value = stats.f_oneway(*[self.data[self.data['PostalCode'] == zip_code]['TotalPremium']
                                     for zip_code in zip_code_groups.index])

        # Interpret the result
        alpha = 0.05
        if p_value < alpha:
            print("Reject the null hypothesis: There are significant margin differences between zip codes.")
        else:
            print("Fail to reject the null hypothesis: No significant margin differences between zip codes.")
    
    def test_risk_differences_between_women_and_men(self):
        # Split data by gender and compute TotalClaims
        gender_groups = [self.data[self.data['Gender'] == 'Female']['TotalClaims'],
                         self.data[self.data['Gender'] == 'Male']['TotalClaims']]
        _, p_value = stats.ttest_ind(*gender_groups, equal_var=False)  # Welch's T-test

        # Interpret the result
        alpha = 0.05
        if p_value < alpha:
            print("Reject the null hypothesis: There are significant risk differences between women and men.")
        else:
            print("Fail to reject the null hypothesis: No significant risk differences between women and men.")
