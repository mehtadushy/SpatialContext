% Computes a stronger(than just relative orientation) context that captures
% both relative orientation and types of SIFTs occurring in the
% neighbourhood of a keypoint through a 2D histogram. The relative
% orientation is pushed into the keypoint orientation, while a coarse
% sift histogram from the neighbourhood is clustered orthogonally and a histogam of SIFT bow computed in each context cluster. It is essentially the same as before, with just a different sort of normalization. The 2D feature is normalized per context cluster before being normalized overall. 

clear;
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'scenes_names.mat'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Flags
%DoG_keypoints = true; % uses DoG keypoints for SIFT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%canSkip_FrameComputation = 0;
%params.axisAlignedSIFT = false;
%%Compute keypoint sift frames
%pfig = sp_progress_bar(sprintf('Computing Keypoint SIFT Frames'));
%if(DoG_keypoints==true)
%    GenerateDoGKeypointSiftFrames( data.filenames,image_dir,data_dir,params,canSkip_FrameComputation,pfig);
%else
%    GenerateKeypointSiftFrames( data.filenames,image_dir,data_dir,params,canSkip_FrameComputation,pfig);
%end
%close(pfig);
%
%%Compute relative orientation context of keypoint and push back into the keypoint orientation
%params.sigmaFactors = [40;
%                       80;
%                       ];
%params.keypointFrameSuffix = '_keypoint_sift_frame.mat';
%
%ComputeRelativeOrientationCorrectedFrame(data.filenames, data_dir, params, canSkip_FrameComputation);
%
%% Now use the frames to compute Keypoint SIFT
%sigmaFactors = params.sigmaFactors;
%
%canSkip_SIFTComputation = 0;
%
%for sf = 1:length(sigmaFactors)
%    %clear params;
%    params.keypointFramePrefix = sprintf('_%d_neigh_ROcontext', sigmaFactors(sf));
%    GenerateKeypointSiftFromFrames(data.filenames,image_dir,data_dir, params, canSkip_SIFTComputation);
%end
%
%
%%%
%% Cluster Keypoint SIFT
%%Pick 1000 images to do k_means on
%canSkip_dictionaryBuild = 0;
%
%clear params;
%dictionarySize = [20, 40, 50, 100, 200, 400, 800, 1000];
%sigmaFactors = [40; 80];
%for sf = 1:length(sigmaFactors)
%   keypointSuffix = sprintf('_%d_neigh_ROcontext_keypoint_sift.mat', sigmaFactors(sf));
%    parfor k = 1:length(dictionarySize)
%        params = struct('numTextonImages', 1000, 'dictionarySize', dictionarySize(k));
%        CalculateDictionary(data.filenames,image_dir,data_dir,keypointSuffix, params, canSkip_dictionaryBuild);
%        BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,keypointSuffix,params, canSkip_dictionaryBuild);
%    end
%end
%%%
%clear params;
%% Create coarse SIFT histograms in keypoint neighbourhoods for use as context
%canSkip_ContextComputation = 0;
%coarseDictionarySize = 20; %[20, 40, 50]; %Coarse Dictionary Size
%sigmaFactors = [40; 80];
%for sf = 1:length(sigmaFactors)
%   keypointSuffix = sprintf('_%d_neigh_ROcontext_keypoint_sift.mat', sigmaFactors(sf));
%    parfor k = 1:length(coarseDictionarySize)
%   	textonSuffix = sprintf('_soft_texton_ind_%d_%d_neigh_ROcontext_keypoint_sift.mat', coarseDictionarySize(k), sigmaFactors(sf));
%        params = struct('dictionarySize', coarseDictionarySize(k),'keypointFeatureSuffix',keypointSuffix, 'sigmaFactor', sigmaFactors(sf), 'scaleWeighted', true);
%        ComputeCoarseNeighbourhoodHistogram(data.filenames,data_dir,textonSuffix, params, canSkip_ContextComputation);
%    end
%end
%clear params;
%%
%% Cluster Keypoint Context
%%Pick 1000 images to do k_means on
%dictionarySize = [10, 20 ,40]; % Context Dictionary Size
%%coarseDictionarySize = [20, 40, 50]; %Coarse Dictionary Size
%neighbourhoodSize = [40; 80];
%parfor k = 1:length(dictionarySize)
% for j = 1:length(coarseDictionarySize)
%    for i = 1:length(neighbourhoodSize)
%    pfig =                     sp_progress_bar(sprintf('Computing Keypoint Context Dictionaries With Dic Size %d',dictionarySize(k)));
%    para = struct('numTextonImages', 1000, 'dictionarySize', dictionarySize(k));
%    CalculateHistogramDictionary(data.filenames,data_dir,sprintf('_%d_neigh_contexthist%d.mat', neighbourhoodSize(i), coarseDictionarySize(j)),para, 0, pfig);
%    BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,sprintf('_%d_neigh_contexthist%d.mat', neighbourhoodSize(i), coarseDictionarySize(j)),para, 0, pfig);
%    close(pfig)
%    end
% end
%end
%
%
%% Next up, we would construct 2D Histograms 
%neighbourhoodSize = [ 40;
%                       80;
%                       ];
%%coarseDictionarySize = [20, 40, 50]; %Coarse Dictionary Size
%contextDictionarySize = [10 ,20, 40];
%keypointDictionarySize = [50, 100, 200, 400,800, 1000];
%config_table = zeros(length(keypointDictionarySize) * length(coarseDictionarySize) * length(contextDictionarySize) * length(neighbourhoodSize), 4);
%idx = 1;
%for j = 1: length(keypointDictionarySize)
%    for l = 1: length(coarseDictionarySize)
%     for k = 1: length(contextDictionarySize)
%      for i = 1:length(neighbourhoodSize)
%      config_table(idx,:) = [keypointDictionarySize(j), coarseDictionarySize(l), contextDictionarySize(k), neighbourhoodSize(i)];
%      idx= idx+1;
%      end
%    end
%  end
%end
%parfor idx = 1: size(config_table,1)
%	    fprintf('Yay yay!');
%            pfig = sp_progress_bar(sprintf('2D Hist of Neigh Size%d Keypt Dic%d Context Dic%d',config_table(idx,4), config_table(idx,1), config_table(idx,3)));
%            CompileNeighborhoodPoolFeature2DSoftAssignHistogramV2( data.filenames,data_dir, sprintf('_soft_texton_ind_%d_%d_neigh_ROcontext_keypoint_sift',config_table(idx,1),config_table(idx,4)), sprintf('_soft_texton_ind_%d_%d_neigh_contexthist%d', config_table(idx,3), config_table(idx,4), config_table(idx,2)), [config_table(idx,1), config_table(idx,3)],[] , 0, pfig );
%            close(pfig)
%end

%
load(fullfile(data_dir2, 'category_names.mat'));
%Do a test-train split (data is already randomized
train_percent = 90;
train.index = (1 : floor(length(data.labels)*train_percent/100));
train.labels = data.labels(train.index);
test.index = (length(train.index)+1 : length(data.labels));
test.labels = data.labels(test.index);
train_test_datafilename = fullfile(data_dir, 'train_test_data.mat');
save(train_test_datafilename, 'train', 'test');

%%
%%%%%%%%%%%%%%%%
% 2D Histograms of keypoint appearance captured with SIFT and keypoint
% context
%%%%%%%%%%%%%%%%
neighbourhoodSize = [ 40;
                       80;
                       ];
coarseDictionarySize = 20;%[20, 40, 50]; %Coarse Dictionary Size
contextDictionarySize = [10 ,20, 40];
keypointDictionarySize = [50, 100, 200, 400,800, 1000];
n_sz = length(neighbourhoodSize);
k_sz = length(keypointDictionarySize);
c_sz = length(contextDictionarySize);

accuracy_results = cell(length(contextDictionarySize)+1, length(neighbourhoodSize)*length(keypointDictionarySize)+1);
config_table = zeros(n_sz*k_sz*c_sz, 3);
idx=1;
for k = 1: length(contextDictionarySize)
     accuracy_results{k+1, 1} = sprintf('%d_Context', contextDictionarySize(k));
     for i = 1:length(neighbourhoodSize)
          for j = 1: length(keypointDictionarySize)
            accuracy_results{1, (i-1)*length(keypointDictionarySize)+j+1} = sprintf('%dN %d_KD', neighbourhoodSize(i), keypointDictionarySize(j));
            config_table(idx,:) = [contextDictionarySize(k),neighbourhoodSize(i), keypointDictionarySize(j)];
            idx= idx+1;
          end      
     end
end

%%


acc_log = cell(1, size(config_table, 1));

parfor i = 1: size(config_table, 1)
            
            inFName = fullfile(data_dir, sprintf('2Dhistograms_%s_with%s.mat', sprintf('_soft_texton_ind_%d_%d_neigh_ROcontext_keypoint_sift',config_table(i,3),config_table(i,2)), sprintf('_soft_texton_ind_%d_%d_neigh_contexthist%d', config_table(i,1), config_table(i,2), coarseDictionarySize)));
            H_all = load(inFName,'-ascii');
            dat = load(train_test_datafilename);
            train = dat.train;
            test = dat.test;
            train_features = H_all(train.index,:);
            test_features = H_all(test.index,:);

            %We'll do multiclass classification
            %Compute kernel for training data
            train.K = [(1:size(train_features,1))' , intersectionKernel(train_features, train_features)];
            test.K = [(1:size(test_features,1))' , intersectionKernel(test_features, train_features)];
    
            bow_model = svmtrain(train.labels',train.K, '-t 4' );
            [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
            acc_log{i} = accuracy_P(1);
end

idx=1;
for k = 1: length(contextDictionarySize)
     for i = 1:length(neighbourhoodSize)
          for j = 1: length(keypointDictionarySize)
            accuracy_results{k+1, (i-1)*length(keypointDictionarySize)+j+1} = acc_log(idx);
            idx= idx+1;
          end      
     end
end



outfname = fullfile( data_dir, 'results', '2DContextHist_accuracy.mat');
save(outfname, 'accuracy_results');




