# -*- coding: utf-8 -*-
"""
Created on 2023
@author: Geon
"""

import pandas as pd
import numpy as np
import os
import difflib

from urllib import parse
from tqdm import tqdm
from datetime import datetime

### 완전 정상적인 웹로그 제외
### 조금이라도 이상하다는 생각이 드는 것 위주로 추출할 수 있도록 진행

domain_url = 'co.kr'

df_1_all = pd.read_csv('./data/1_preprocessed/01_second/D_01/' + domain_url + '_2_url.csv', encoding = 'utf-8')
del(df_1_all['Unnamed: 0'])

# 도메인 명이 없는 경우 제외
df_2_all = df_1_all[df_1_all['url'].str.contains(domain_url) == True]

df_2_all['domain'] = df_2_all['url'].str.split('/', 1).str[0]
df_2_all['full_url'] = df_2_all['url'].str.split('/', 1).str[1]

# 도메인 명이 없는 경우
no_0_all = df_1_all[df_1_all['url'].str.contains(domain_url) == False]


##### URL PATH 관련 처리 #####
# path 단계
df_2_all['path_cnt'] = 0

# 파라미터 유무
df_2_all['param_cnt'] = df_2_all['full_url'].str.count('\?')

# ?가 많은 경우 (인코딩이 잘못 되었을 경우 물음표(?)가 많을 수 있음)
no_1_all = df_2_all[df_2_all['param_cnt'] > 1]

# ?가 많은 경우 제외
df_3_all = df_2_all[df_2_all['param_cnt'] < 2]

df_3_all['url_path'] = df_3_all['full_url'].str.split('#', 1).str[0]        # URL_PATH
df_3_all['fragment'] = df_3_all['full_url'].str.split('#', 1).str[1]        # Fragment 아이디 (Fragment 형태 : #id)

df_3_all['url_path'] = df_3_all['url_path'].str.split('?', 1).str[0]        # URL_PATH
df_3_all['parameter'] = df_3_all['full_url'].str.split('?', 1).str[1]       # Parameter (파라미터 형태 : key=value&key=value)


df_3_all['jsession'] = df_3_all['url_path'].str.split(';', 1).str[1]        # JessionID (jession:ajsdf29vjckdf)
df_3_all['url_path'] = df_3_all['url_path'].str.split(';', 1).str[0]        # URL_PATH (파라미터, jession을 제외한 웹경로 + 파일명)

df_3_all['url_path'] = df_3_all['url_path'].str.replace(r'/$', '')          # 맨마지막 문자가 '/' 인 경우 제외
df_3_all['path_cnt'] = df_3_all['url_path'].str.count('/')                  # 문자 '/' 카운트
df_3_all['path_only'] = ''

df_3_all['file_name'] = df_3_all['url_path'].str.rsplit('/', 1).str[1]      # 파일명 찾기 (오른쪽 기준으로 '/' split)
df_3_all['file_name'][df_3_all['path_cnt'] == 0] = df_3_all['url_path']     # 파일경로 가 없을 경우 url_path가 파일명임
df_3_all['file_ext'] = ''

df_3_all['file_dot_cnt'] = df_3_all['file_name'].str.count('\.')            # 파일명의 확장자 찾기 위한 문자 '.' 카운트

df_3_all['url_path'][df_3_all['file_dot_cnt'] == 0] = df_3_all['url_path'][df_3_all['file_dot_cnt'] == 0] + '/'  # 파일명이 아닐 경우 파일경로에 마지막 부분에 문자 '/' 붙이기
df_3_all['file_name'][df_3_all['file_dot_cnt'] == 0] = ''
df_3_all['path_cnt'] = df_3_all['url_path'].str.count('/')                  # 파일 경로 깊이 카운트

df_3_all['file_ext'] = df_3_all['file_name'].str.rsplit('.', 1).str[1]      # 파일 확장자 추출
df_3_all = df_3_all.reset_index(drop=True)                                  # index 초기화


ext_0_all = df_3_all.loc[:, ['file_ext', 'cnt']]


ext_0_all['file_ext'] = ext_0_all['file_ext'].str.lower()
ext_1_all = ext_0_all.groupby(['file_ext']).sum()
ext_1_all = ext_1_all.reset_index(drop=False)
ext_1_all = ext_1_all.sort_values(by=['cnt', 'file_ext'], ascending=[False, True])            # 데이터프레임 정렬
ext_1_all = ext_1_all.reset_index(drop=True)

print('-------------------------------')
print(ext_1_all['cnt'].sum())
print('-------------------------------')
ext_1_all['percent'] = ext_1_all['cnt'] / ext_1_all['cnt'].sum() * 100

### url path 깊이에 따른 동적 컬럼 생성 관련 부분
path_cnt_df = df_3_all['path_cnt'].to_frame()                               # 컬럼 한개 추출 후 DataFrame화
path_cnt_df = path_cnt_df.drop_duplicates(['path_cnt'], keep='first')       # 중복제거
path_cnt_df = path_cnt_df.sort_values(by=['path_cnt'], ascending=[True])    # 데이터프레임 정렬
path_cnt_df = path_cnt_df.reset_index(drop=True)                            # index 초기화

full_size = path_cnt_df.loc[len(path_cnt_df)-1, 'path_cnt']
str_column = []

for i in range(0, full_size):
    df_3_all['path_' + str(i)] = ''
    str_column.append('path_' + str(i))

for chk_idx in range(0, len(path_cnt_df)):
    one_chk = path_cnt_df.loc[chk_idx, :]                   # path_cnt_df (path 단계 관련)
    
    num_chk = one_chk['path_cnt']
    
    if num_chk > 0:
        # 0보단 큰 경우만 진행
        for i in range(0, num_chk):
        
            temp_col = 'path_' + str(i)
            df_3_all[temp_col][df_3_all['path_cnt'] == num_chk] = df_3_all['url_path'].str.split('/').str[i]
        
            df_3_all['path_only'][df_3_all['path_cnt'] == num_chk] = df_3_all['path_only'] + '/' + df_3_all['url_path'].str.split('/').str[i]


df_3_all['path_only'][df_3_all['path_only'] == '/'] = ''
df_3_all['domain'] = domain_url

no_2_all = df_3_all[df_3_all['file_name'].str.contains('wp[a-zA-Z0-9_-]+\.php') == True]
no_2_all = no_2_all.append(df_3_all[df_3_all['file_name'].str.contains('xml[a-zA-Z0-9_-]+\.php') == True])


df_4_all = df_3_all[df_3_all['file_name'].str.contains('wp[a-zA-Z0-9_-]+\.php') != True]
df_4_all = df_4_all[df_4_all['file_name'].str.contains('xml[a-zA-Z0-9_-]+\.php') != True]

df_4_all = df_4_all.reset_index(drop=True)                                  # index 초기화


new_column = []
new_column.extend(str_column)
new_column.append('url')
new_column.append('cnt')
new_column.append('domain')
new_column.append('url_path')
new_column.append('parameter')
new_column.append('path_only')
new_column.append('file_name')
new_column.append('file_ext')

tot_0_all = df_4_all.loc[:, new_column]

### 정리기준 1 >> url 경로 + 파일명 기준으로 카운트
dul_0_column = []
dul_0_column.append('domain')
dul_0_column.append('url_path')
dul_0_column.extend(str_column)
dul_0_column.append('file_name')
dul_0_column.append('file_ext')

tot_1_all = tot_0_all.groupby(dul_0_column).sum()
tot_1_all = tot_1_all.reset_index(drop=False)
tot_1_all = tot_1_all.sort_values(by=['cnt', 'url_path'], ascending=[False, True])            # 데이터프레임 정렬
tot_1_all = tot_1_all.reset_index(drop=True)

### 정리기준 2 >> url 경로 기준으로만 칸운트
dul_1_column = []
dul_1_column.append('domain')
dul_1_column.append('path_only')
dul_1_column.extend(str_column)

tot_2_all = tot_0_all.groupby(dul_1_column).sum()
tot_2_all = tot_2_all.reset_index(drop=False)
tot_2_all = tot_2_all.sort_values(by=['cnt', 'path_only'], ascending=[False, True])    # 데이터프레임 정렬
tot_2_all = tot_2_all.reset_index(drop=True)


str_g_column = []

for column in str_column:
    print(column)
    grp_column = column + '_g'
    len_column = column + '_l'
    
    str_g_column.append(grp_column)
    
    tot_2_all[grp_column] = tot_2_all[column]
    tot_2_all[len_column] = tot_2_all[column].str.len()
    tot_2_all[len_column] = tot_2_all[len_column].apply(str)
    
    tot_2_all[grp_column][tot_2_all[column].str.contains('^[a-zA-Z0-9_-]+$') == True] = 'A_N_U_' + tot_2_all[len_column][tot_2_all[column].str.contains('^[a-zA-Z0-9_-]+$') == True]
    tot_2_all[grp_column][tot_2_all[column].str.contains('^[a-zA-Z0-9,]+$') == True] = 'A_N_C_' + tot_2_all[len_column][tot_2_all[column].str.contains('^[a-zA-Z0-9,]+$') == True]
    tot_2_all[grp_column][tot_2_all[column].str.contains('^[a-zA-Z0-9]+$') == True] = 'A_N_' + tot_2_all[len_column][tot_2_all[column].str.contains('^[a-zA-Z0-9]+$') == True]
    tot_2_all[grp_column][tot_2_all[column].str.contains('^[a-zA-Z]+$') == True] = tot_2_all[column][tot_2_all[column].str.contains('^[a-zA-Z]+$') == True]
    tot_2_all[grp_column][tot_2_all[column].str.contains('^[0-9]+$') == True] = 'N_' + tot_2_all[len_column][tot_2_all[column].str.contains('^[0-9]+$') == True]
    
    
    #tot_2_all[grp_column] = tot_2_all[grp_column] + str(tot_2_all[len_column])

dul_2_column = []
dul_2_column.append('domain') 
dul_2_column.append('cnt') 
dul_2_column.extend(str_g_column)

tot_2_1_all = tot_2_all.loc[:, dul_2_column]
tot_2_2_all = tot_2_1_all.groupby(str_g_column).sum()
tot_2_2_all = tot_2_2_all.reset_index(drop=False)
tot_2_2_all = tot_2_2_all.sort_values(by=str_g_column)    # 데이터프레임 정렬
tot_2_2_all = tot_2_2_all.reset_index(drop=True)

########################################################################

param_column = []
param_column.append('url')
param_column.append('cnt')
param_column.append('domain')
param_column.append('full_url')
param_column.append('param_cnt')
param_column.append('url_path')
param_column.append('file_name')
param_column.append('parameter')

param_0_all = df_4_all.loc[:, param_column]
param_1_all = param_0_all[param_0_all['param_cnt'] == 1]

param_1_all['p_and_cnt'] = param_1_all['parameter'].str.count('&')

### 파라미터 key, value 개수
param_cnt_df = param_1_all['p_and_cnt'].to_frame()
param_cnt_df = param_cnt_df.drop_duplicates(['p_and_cnt'], keep='first')       # 중복제거
param_cnt_df = param_cnt_df.sort_values(by=['p_and_cnt'], ascending=[True])    # 데이터프레임 정렬
param_cnt_df = param_cnt_df.reset_index(drop=True)                             # index 초기화

param_max_size = param_cnt_df.loc[len(param_cnt_df)-1, 'p_and_cnt'] + 1

par_column = []

for i in range(0, param_max_size):
    #param_1_all['param_' + str(i)] = np.nan
    param_1_all['param_' + str(i)] = ''
    par_column.append('param_' + str(i))


for chk_idx in range(0, len(param_cnt_df)):
    one_chk = param_cnt_df.loc[chk_idx, :]                   # path_cnt_df (path 단계 관련)
    
    num_chk = one_chk['p_and_cnt']
    
    for i in range(0, num_chk + 1):
        
        temp_col = 'param_' + str(i)
        param_1_all[temp_col][param_1_all['p_and_cnt'] == num_chk] = param_1_all['parameter'].str.split('&').str[i]
        
param_2_all = pd.DataFrame(columns = ['domain', 'file_name', 'param_data', 'cnt'])


for column in par_column:
    
    temp_column = []
    temp_column.append('domain')
    temp_column.append('file_name')
    temp_column.append(column)
    temp_column.append('cnt')
    
    temp_df = param_1_all[temp_column]
    temp_df.rename(columns={column:"param_data"}, inplace = True)
    
    param_2_all = pd.concat([param_2_all, temp_df])


param_3_all = param_2_all.dropna(axis=0)
param_3_all = param_3_all[param_3_all['param_data'] != '']

param_3_all['cnt'] = pd.to_numeric(param_3_all['cnt'])
param_3_all['param_key'] = param_3_all['param_data'].str.split('=', 1).str[0]
param_3_all['param_value'] = param_3_all['param_data'].str.split('=', 1).str[1]

param_4_all = param_3_all.copy()
param_4_all = param_4_all.dropna(axis=0)

### 정리기준 3 >> 파라미터 key 값으로 정리
tot_3_all = param_3_all.groupby('param_key').sum()
tot_3_all = tot_3_all.reset_index(drop=False)
tot_3_all = tot_3_all.sort_values(by=['cnt', 'param_key'], ascending=[False, True])    # 데이터프레임 정렬
tot_3_all = tot_3_all.reset_index(drop=True)


### 정리기준 4 >> 파라미터 value 값으로 정리
tot_4_all = param_3_all.groupby('param_value').sum()
tot_4_all = tot_4_all.reset_index(drop=False)
tot_4_all = tot_4_all.sort_values(by=['cnt', 'param_value'], ascending=[False, True])    # 데이터프레임 정렬
tot_4_all = tot_4_all.reset_index(drop=True)

tot_4_all['param_check'] = ''

tot_4_all['param_check'][tot_4_all['param_value'].str.contains('from') == True] = 'SQL_INJECTION_FROM'
tot_4_all['param_check'][tot_4_all['param_value'].str.contains('select') == True] = 'SQL_INJECTION_SELECT'
tot_4_all['param_check'][tot_4_all['param_value'].str.contains('insert') == True] = 'SQL_INJECTION_INSERT'
tot_4_all['param_check'][tot_4_all['param_value'].str.contains('update') == True] = 'SQL_INJECTION_UPDATE'
tot_4_all['param_check'][tot_4_all['param_value'].str.contains('delete') == True] = 'SQL_INJECTION_DELETE'

tot_4_all['url_clear'] = tot_4_all['param_value'].apply(parse.unquote)
tot_4_all['url_clear2'] = tot_4_all['url_clear'].apply(parse.unquote)

param_4_all = param_4_all.sort_values(by=['file_name', 'param_data', 'cnt'], ascending=[True, True, False])    # 데이터프레임 정렬
param_4_all['cnt'] = pd.to_numeric(param_4_all['cnt'])
param_4_all
param_4_all = param_4_all.reset_index(drop=True)



param_4_all['param_key'][param_4_all['param_key'].str.contains('^amp;') == True] = param_4_all['param_key'].str.replace('^amp;', '')

param_4_all['param_key_len'] = param_4_all['param_key'].str.len()
param_4_all['param_key_grp'] = ''

param_4_all['param_key_grp'][param_4_all['param_key'].str.contains('^[a-zA-Z0-9_-]+$') == True] = 'ALPHA_N'
param_4_all['param_key_grp'][param_4_all['param_key'].str.contains('^[a-zA-Z_]+$') == True] = 'ALPHA_U'
param_4_all['param_key_grp'][param_4_all['param_key'].str.contains('^[a-zA-Z\.]+$') == True] = 'ALPHA_D'
param_4_all['param_key_grp'][param_4_all['param_key'].str.contains('^[a-zA-Z-]+$') == True] = 'ALPHA_H'
param_4_all['param_key_grp'][param_4_all['param_key'].str.contains('^[a-zA-Z;]+$') == True] = 'ALPHA_S'
param_4_all['param_key_grp'][param_4_all['param_key'].str.contains('^[a-zA-Z]+$') == True] = 'ALPHA'


param_4_all['param_key_grp'][param_4_all['param_key'].str.contains('^[0-9]+\.[0.9]+$') == True] = 'FLOAT'
param_4_all['param_key_grp'][param_4_all['param_key'].str.contains('^[0-9]+$') == True] = 'NUM'

param_4_all['param_value2'] = param_4_all['param_value']
param_4_all['param_value3'] = ''
param_4_all['param_value_grp'] = ''

param_4_all['param_value_grp'][param_4_all['param_value2'].str.contains('%[a-zA-Z0-9]{2}')] = 'ENCSTR'
param_4_all['param_value_grp'][param_4_all['param_value2'].str.contains('%25[a-zA-Z0-9]{2}')] = 'ENCSTR_2'

param_4_all['param_value2'][param_4_all['param_value_grp'] == 'ENCSTR_2'] = param_4_all['param_value'].str.replace('%25', '%')
param_4_all['param_value3'] = param_4_all['param_value2'].apply(lambda x : parse.unquote(x))

#param_4_all['param_value_grp'][param_4_all['param_key'].str.contains('((%\w{2}){2,}|(%2\w)+|(%[357][A-E])+|(%[46]0)+)') == True] = 'ENCSTR'


"""
#param_4_all['param_value2'][[param_4_all['param_value'].str.contains('%25[0-9a-zA-Z]{2}') == True]] = param_4_all['param_value'].str.replace('%25', '%')
param_4_all['param_value2'] = param_4_all['param_value'].str.replace('%25', '%')
param_4_all['param_value3'] = param_4_all['param_value2'].apply(lambda x : parse.unquote(x))
"""


key_0_all = param_4_all.loc[:,['domain', 'file_name', 'cnt', 'param_key', 'param_key_grp', 'param_key_len', 'param_value']]
key_0_all['cnt'] = pd.to_numeric(key_0_all['cnt'])
key_1_all = key_0_all.groupby(['param_key_grp', 'param_key_len', 'param_key', 'file_name']).sum()


