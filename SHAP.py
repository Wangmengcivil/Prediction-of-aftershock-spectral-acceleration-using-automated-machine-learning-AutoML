# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:32:04 2022

@author: davince
"""

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
regmain = joblib.load('D:\\科研文件\\20220623 谱相容余震地震动生成\\database\\SaY.pkl')

data = pd.read_csv('D:\\科研文件\\20220623 谱相容余震地震动生成\\database\\inputtest.csv')

x = data.iloc[:, :4]
y = data.iloc[:, 4]



# scaler0 = MinMaxScaler() 
# scaler0.fit(x)
# x = scaler0.transform(x)
x = pd.DataFrame(x)
x.columns = ['$\it{M}$$_\mathrm{w}$','Hypocenter distance','$\it{V}$$_\mathrm{s}30$','$\it{S}$$_\mathrm{a}$(Mainshock)']

explainer = shap.Explainer(regmain)
shap_values = explainer(x)
shap_interaction_values = shap.TreeExplainer(regmain).shap_interaction_values(x)
shap_values0 = explainer.shap_values(x)
#with plt.style.context('science','nature'):
#    shap.force_plot(shap_values[0], x.iloc[0], matplotlib=True)
#    shap.force_plot(shap_values[10], x.iloc[10], matplotlib=True)
#    shap.force_plot(shap_values[20], x.iloc[20], matplotlib=True)
#    shap.force_plot(shap_values[100], x.iloc[100], matplotlib=True)
#    shap.force_plot(shap_values[200], x.iloc[200], matplotlib=True)

#shap.plots.heatmap(shap_values)

#shap.plots.beeswarm(shap_values)
#shap.summary_plot(shap_interaction_values, x)
#shap.plots.bar(shap_values)

with plt.style.context('science','nature'):
    shap.plots.bar(shap_values)


with plt.style.context('science','nature'):
 shap.dependence_plot ('$\it{M}$$_\mathrm{w}$', shap_values0, x, display_features=x, interaction_index='auto', dot_size=26)
 shap.dependence_plot ('Hypocenter distance', shap_values0, x, display_features=x, interaction_index='auto', dot_size=26)
 shap.dependence_plot ('$\it{V}$$_\mathrm{s}30$', shap_values0, x, display_features=x, interaction_index='auto', dot_size=26)
 shap.dependence_plot ('$\it{S}$$_\mathrm{a}$(Mainshock)', shap_values0, x, display_features=x, interaction_index='auto', dot_size=26)
