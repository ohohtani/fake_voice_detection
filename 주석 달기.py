import numpy as np
import librosa                                    # 오디오 분석 라이브러리
from sklearn.mixture import GaussianMixture       # 가우시안 혼합모델(머신러닝에서 사용)
import matplotlib                                 # 그래프를 그리기 위함
matplotlib.use('Agg')                             
import matplotlib.pyplot as plt
import pickle                                     # 파이썬 객체 직렬화/역직렬화 (?)
import argparse                                   # 명령줄 인수를 파싱(?)
import os                                         # 파일 및 디렉토리 조작
import pandas as pd


def load_wavs_as_matrices(data_dir):                  # 지정한 디렉토리의 오디오 파일을 읽어 MFCC(각 파일의 델 주파수 켑스트럼 계수) 추출  ->  쉽게 설명 : 소리를 컴퓨터가 이해하기 쉽게 변환
    filenames = os.listdir(data_dir)                  # 파일지정
    out = []                                          # 아웃풋 리스트 선언
    for filename in filenames:                        # 파일을 모두 순회하며
        filepath = os.path.join(data_dir,filename)    # 각 파일이름(filename)을 data_dir 경로와 결합하여 전체경로 filepath를 생성
        y, sr = librosa.load(filepath)                # librosa 라이브러리의 load 함수를 사용하여 오디오 파일을 읽음. y->오디오 신호 데이터 , sr->초당 샘플 수
        mfccs=librosa.feature.mfcc(y,sr)              # librosa의 mfcc 함수를 사용해 MFCC 특징 추출.
        out.append(mfccs.T)                           # 아웃풋 리스트에 MFCC 특징을 추가 (전치행렬 쓰는 이유?)
    return out


def load_datasets():                                                  # 학습 데이터 & 테스트 데이터 불러오는 함수
    # load training data and test data 
    train_data_dirs = {                                               # 학습 데이터의 경로 (verif-대상 검증, ubg-백그라운드 검증)
        'Verif_disjoint':'./data/target/train_verification/',
        'ubg'           :'./data/ubg/train_verification/'
    }

    train_data = {}                                                   # 학습 데이터 저장할 딕셔너리
    for name, data_dir in train_data_dirs.items():                    # 파일을 순회하며 name(key), data_dir(value)값을 가져옴.
        train_data[name] = load_wavs_as_matrices(data_dir)            # data_dir에서 데이터를 로드하고 train_data 딕셔너리에 추가
    
    test_data_dirs = {                                                # 테스트 데이터 경로
        'train_Conv' :'./data/target/train_conversion/',              # 변환용 학습 데이터
        'test'       :'./data/target/test/',                          # 대상 테스트 데이터
        'ubg_test'   :'./data/ubg/test/',                             # 백그라운드 테스트 데이터
        'fake'       :'./data/fake'                                   # 페이크 데이
    }

    test_data = {}                                                    # 테스트 데이터 저장할 딕셔너리
    for name, data_dir in test_data_dirs.items():                     
        test_data[name] =load_wavs_as_matrices(data_dir)
        
    # make shared data                                                        # 공유 데이터를 생성할거임
    test_data['validation_Verif'] = train_data['Verif_disjoint'][-20:]        # 마지막 20개의 항목 test_data에
    train_data['Verif_disjoint']  = train_data['Verif_disjoint'][:-20]        # 마지막 20개 제외하고 전부 train_data에
    train_data['Verif_shared'] = train_data['Verif_disjoint']+test_data['train_Conv'][-20:] # 다른 train_data에는 결합해서 저장
    test_data['train_Conv'] = test_data['train_Conv'][-20:]                   # 또다른 test_data 파일.. 
    return train_data,test_data                                               # 일단은 뭐 훈련에 많이, 테스트에 적게 라는건 알겠다.


if __name__ == '__main__':        # 데이터 로드하는 거고 모델 쪽으로 중요한 건 없으니까 상세분석은 미뤄도 될 거 같다다
    
    # get model directory
    parser = argparse.ArgumentParser(description = 'Train GMM models for verification and plot the scores for data.')
    model_dir_default = './model/verification_gmm/pretrained'
    parser.add_argument('--model_dir', type = str, help = 'Directory for the pre-trained model.', default = model_dir_default)
    argv = parser.parse_args()
    model_dir = argv.model_dir
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    #load data
    print('loading data ... ')
    train_data, test_data = load_datasets()
    print('Done.')

    ##############################################
    # train gmm and plot for 2 cases:
    # 1. disjoint data for VC and Verification
    # 2. shared data for VC and Verification 
    ##############################################

    # load pretraind gmm ubg model if it exists
    model_path = './model/verification_gmm/pretrained/gmm_ubg.p'
    if os.path.exists(model_path):
        gmm_ubg = pickle.load(open( model_path, "rb" ))    
    else:
        print('training GMM-UBG model ... '.format(case), end='')
        fit_data = np.concatenate(train_data['ubg'])
        gmm_ubg = GaussianMixture(n_components=2048, covariance_type='diag').fit(fit_data)
        pickle.dump(gmm_ubg, open( model_path, "wb" ), protocol=pickle.HIGHEST_PROTOCOL)
        print('Done.')


    for case in ['disjoint','shared']:

        # load pretrained model if it exists
        model_path = './model/verification_gmm/pretrained/gmm_target_({}_data_for_VC_and_Verif).p'.format(case)
        if os.path.exists(model_path):
            gmm_target = pickle.load(open( model_path, "rb" ))
        else:
            print('training GMM speaker model for {} case ... '.format(case))
            fit_data = np.concatenate([ clip_data for clip_data in train_data['Verif_{}'.format(case)] ])
            gmm_target = GaussianMixture(n_components=256, covariance_type='diag').fit(fit_data)
            pickle.dump(gmm_target, open( model_path, "wb" ), protocol=pickle.HIGHEST_PROTOCOL)
            print('Done.')

        # log likelihood ratio    
        def get_LR(samples):
            return + gmm_target.score_samples(samples) - gmm_ubg.score_samples(samples)


        # plot from here
        # for plotting
        names = ['test','ubg_test','fake','train_Conv','validation_Verif']
        if not os.path.exists('./out/plot/'):
            os.makedirs('./out/plot/')

        
        # plot score for small clip (2~10 sec)
        i=0
        one_clip_scores = {name:get_LR(test_data[name][i]) for name in ['test','ubg_test','fake']}
        
        plt.figure(figsize=[10,6])
        for name in one_clip_scores.keys():
            plt.hist(one_clip_scores[name], alpha=0.5, bins=50, density=True,range=[-50,50])
        plt.legend(['test','universal background', 'fake'])
        plt.savefig('./out/plot/score_for_one_small_clip_({}_data_for_VC_and_Verif).png'.format(case), dpi=1000)
        
        # print mean for each test dataset
        print('average scores of {} data for VC and Verif'.format(case))
        for name in names:
            print(name+':'+' '*(19-len(name))+str(np.mean([get_LR(data).mean() for data in test_data[name]]))) 
        
        # plot score for whole data
        scores = {name: np.concatenate([get_LR(data) for data in test_data[name]])
            for name in names} 
        
        plt.figure(figsize=[10,6])
        for name in names:
            plt.hist(scores[name], alpha=0.5, bins=50, density=True,range=[-50,50])
        plt.legend(['test','universal background', 'fake','train_conversion','validation_verification'])
        plt.savefig('./out/plot/score_for_whole_({}_data_for_VC_and_Verif).png'.format(case), dpi=1000)
        
        # plot average score per small clip for whole data
        score_means = {name: np.array([get_LR(data).mean() for data in test_data[name]])
            for name in names}
        
        plt.figure(figsize=[10,6])
        for name in names:
            plt.hist(score_means[name], alpha=0.5, bins=50, density=True,range=[-20,20])
        plt.legend(['test','universal background', 'fake','train_conversion','validation_verification'])
        plt.savefig('./out/plot/average_score_per_small_clip_for_whole_({}_data_for_VC_and_Verif).png'.format(case), dpi=1000)
       
        if not os.path.exists('./out/scores/'):
            os.makedirs('./out/scores/')

        # save statistics for explatory analysis
        pickle.dump(one_clip_scores, 
                    open( './out/scores/one_clip_scores_({}_data_for_VC_and_Verif).p'.format(case), "wb" ), 
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(scores, 
                    open( './out/scores/scores_({}_data_for_VC_and_Verif).p'.format(case), "wb" ), 
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(score_means, 
                    open( './out/scores/score_means_({}_data_for_VC_and_Verif).p'.format(case), "wb" ), 
                    protocol=pickle.HIGHEST_PROTOCOL)

    # just plot the histgram used in report
    for case in ['disjoint']:
        score_means = pickle.load(open( './out/scores/score_means_({}_data_for_VC_and_Verif).p'.format(case), 'rb') )
        plt.figure(figsize=[10,6])
        i = 1
        for name in ['test','fake','ubg_test']:
            plt.hist(score_means[name], alpha=0.5, bins=70, density=True,range=[-20,10])
        plt.legend(['test','fake','universal background'])
        plt.savefig('./out/plot/average_score_per_small_clip_for_test,fake,ubg.svg', format="svg")        
