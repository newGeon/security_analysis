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
df_3_all['file_ext'] = df_3_all['file_ext'].str.lower()
df_3_all = df_3_all.reset_index(drop=True)                                  # index 초기화

df_3_all['check'] = False

true_0_all = df_3_all[df_3_all['param_cnt'] == 0]

#file_ext = ['png', 'svg', 'gif', 'bmp', 'jpeg', 'ico', 'jpg', 'css', 'js', 'woff', 'eot']
file_ext = ['bmp', 'rle', 'dib', 'jpg', 'jpeg', 'gif', 'png', 'tif', 'tiff', 'svg', 'ico', 'mp3', 'mp4', 'avi', 'mkv', 'wmv',
            'woff', 'woff2', 'eot', 'otf', 'ttf', 'pdf', 'ppt', 'pptx', 'doc', 'docx', 'xls', 'xlsx', 'hwp', 'css', 'js']

# 가장 많이 사용되는 웹서비스 확장자 (정리 한 번 더 필요)
service_ext = ['do', 'jsp', 'html', 'php', 'asp', 'aspx']

true_1_all = pd.DataFrame()

print('=========================================================')
for one in file_ext:
    
    print(one)
    print('-----------------------------------------------------------')
    df_3_all['check'][(df_3_all['param_cnt'] == 0) & (df_3_all['file_ext'] == one)] = True

print('=========================================================')
test_0_all = df_3_all[df_3_all['check'] == True]


ext_0_all = df_3_all.loc[:, ['file_ext', 'cnt']]


ext_0_all['file_ext'] = ext_0_all['file_ext'].str.lower()
ext_1_all = ext_0_all.groupby(['file_ext']).sum()
ext_1_all = ext_1_all.reset_index(drop=False)
ext_1_all = ext_1_all.sort_values(by=['cnt', 'file_ext'], ascending=[False, True])            # 데이터프레임 정렬
ext_1_all = ext_1_all.reset_index(drop=True)


bs_a_0_df = df_3_all[df_3_all['param_cnt'] == 0]

bs_b_0_df = df_3_all[df_3_all['param_cnt'] == 1]


bs_b_0_df['p_and_cnt'] = 0
bs_b_0_df['p_and_cnt'][bs_b_0_df['param_cnt'] == 1] = bs_b_0_df['parameter'][bs_b_0_df['param_cnt'] == 1].str.count('&')



### 파라미터 key, value 개수
param_cnt_df = bs_b_0_df['p_and_cnt'].to_frame()
param_cnt_df = param_cnt_df.drop_duplicates(['p_and_cnt'], keep='first')       # 중복제거
param_cnt_df = param_cnt_df.sort_values(by=['p_and_cnt'], ascending=[True])    # 데이터프레임 정렬
param_cnt_df = param_cnt_df.reset_index(drop=True)                             # index 초기화

param_max_size = param_cnt_df.loc[len(param_cnt_df)-1, 'p_and_cnt'] + 1

par_column = []

for i in range(0, param_max_size):
    #param_1_all['param_' + str(i)] = np.nan
    bs_b_0_df['param_' + str(i)] = ''
    bs_b_0_df['param_k_' + str(i)] = ''
    bs_b_0_df['param_k_g_' + str(i)] = ''
    bs_b_0_df['param_v_' + str(i)] = ''
    bs_b_0_df['param_v_g_' + str(i)] = ''
    
    par_column.append('param_' + str(i))
    par_column.append('param_k_' + str(i))
    par_column.append('param_k_g_' + str(i))
    par_column.append('param_v_' + str(i))
    par_column.append('param_v_g_' + str(i))


for chk_idx in tqdm(range(0, len(param_cnt_df))):
    
    one_chk = param_cnt_df.loc[chk_idx, :]                   # path_cnt_df (path 단계 관련)
    
    num_chk = one_chk['p_and_cnt']
    
    for i in range(0, num_chk + 1):
        
        temp_col = 'param_' + str(i)
        temp_col_k = 'param_k_' + str(i)
        temp_col_k_g = 'param_k_g_' + str(i)
        temp_col_v = 'param_v_' + str(i)
        temp_col_v_g = 'param_v_g_' + str(i)
        
        temp_all = bs_b_0_df['parameter'].str.split('&').str[i]
        temp_key = temp_all.str.split('=').str[0]
        temp_val = temp_all.str.split('=').str[1]
        
        bs_b_0_df[temp_col][bs_b_0_df['p_and_cnt'] == num_chk] = temp_all
        bs_b_0_df[temp_col_k][bs_b_0_df['p_and_cnt'] == num_chk] = temp_key
        bs_b_0_df[temp_col_k_g][bs_b_0_df[temp_col_k].str.contains('^[a-zA-Z0-9_-]+$') == True] = 'ALPHA_N'
        
        bs_b_0_df[temp_col_v][bs_b_0_df['p_and_cnt'] == num_chk] = temp_val
        bs_b_0_df[temp_col_v_g][bs_b_0_df[temp_col_v].str.contains('^[a-zA-Z0-9_-]+$') == True] = 'ALPHA_N'


"""
df_3_all['param_0'][(df_3_all['param_cnt'] == 0) & (df_3_all['p_and_cnt'] == 0)] = ''
df_3_all['param_k_0'][(df_3_all['param_cnt'] == 0) & (df_3_all['p_and_cnt'] == 0)] = ''
df_3_all['param_v_0'][(df_3_all['param_cnt'] == 0) & (df_3_all['p_and_cnt'] == 0)] = ''
"""

bs_b_0_df['check_cnt'] = 0
bs_b_0_df['check_tf'] = False

for i in range(0, param_max_size):
    
    temp_col = 'param_' + str(i)
    temp_col_k = 'param_k_' + str(i)
    temp_col_k_g = 'param_k_g_' + str(i)
    temp_col_v = 'param_v_' + str(i)
    temp_col_v_g = 'param_v_g_' + str(i)
    
    bs_b_0_df['check_cnt'][(bs_b_0_df[temp_col_k] != '') & (bs_b_0_df[temp_col_k_g] == 'ALPHA_N') & 
                           (bs_b_0_df[temp_col_v] != '') & (bs_b_0_df[temp_col_v_g] == 'ALPHA_N')] = bs_b_0_df['check_cnt'][(bs_b_0_df[temp_col_k] != '') & 
                                                                                                                            (bs_b_0_df[temp_col_k_g] == 'ALPHA_N') & 
                                                                                                                            (bs_b_0_df[temp_col_v] != '') & 
                                                                                                                            (bs_b_0_df[temp_col_v_g] == 'ALPHA_N')] + 1
    
bs_b_0_df['check_tf'][bs_b_0_df['check_cnt'] == (bs_b_0_df['p_and_cnt'] + 1)] = True
bs_b_0_df['check_tf'][(bs_b_0_df['param_cnt'] == 1) & (bs_b_0_df['parameter'] == '') & (bs_b_0_df['check'] == True)] = True

bs_b_1_df = bs_b_0_df.loc[:, ['url', 'check_tf']]

df_3_all = pd.merge(df_3_all, bs_b_1_df, on='url', how='left')
df_3_all['check'][df_3_all['check_tf'] == True] = True

test_1_all = df_3_all[df_3_all['check'] == True]
