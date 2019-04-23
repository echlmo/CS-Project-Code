from scipy import stats as st

ctrl = [67,67,67,67,67]
g = [55,60,61,55,59]
a = [30,36,39,36,34]
d = [37,33,30,35,37]

glaucoma_res = st.ttest_ind(ctrl,g,equal_var=False)
amd_res = st.ttest_ind(ctrl,a,equal_var=False)
dr_res = st.ttest_ind(ctrl,d,equal_var=False)

print(glaucoma_res,amd_res,dr_res)
