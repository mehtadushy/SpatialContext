% Compute SIFT with the regular magnification of scale and then at the same
% locations with a larger magnification of scales, and use the latter as
% context
clear;
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'scenes_names.mat'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Flags
DoG_keypoints = true; % uses DoG keypoints for SIFT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
canSkip_FrameComputation =0;

params.axisAlignedSIFT = false;
%Compute keypoint sift frames
if(DoG_keypoints==true)
   GenerateDoGKeypointSiftFrames( data.filenames,image_dir,data_dir,params,canSkip_FrameComputation);
else
   GenerateKeypointSiftFrames( data.filenames,image_dir,data_dir,params,canSkip_FrameComputation);
end

%Compute relative orientation context of keypoint and push back into the keypoint orientation
params.sigmaFactors = [%40;
                      80;
                      ];
params.keypointFrameSuffix = '_keypoint_sift_frame.mat';

ComputeRelativeOrientationCorrectedFrame(data.filenames, data_dir, params, canSkip_FrameComputation);

% Now use the frames to compute Keypoint SIFT
sigmaFactors = params.sigmaFactors;

canSkip_SIFTComputation = 0;

for sf = 1:length(sigmaFactors)
   %clear params;
   params.keypointFramePrefix = sprintf('_%d_neigh_ROcontext', sigmaFactors(sf));
   GenerateCoarseKeypointSiftFromFrames(data.filenames,image_dir,data_dir, params, canSkip_SIFTComputation);
end


%%
% Cluster Keypoint SIFT
%Pick 1000 images to do k_means on
canSkip_dictionaryBuild = 0;

clear params;
dictionarySize = [%20, 40, 50, 100, 
                200, 400, 800, 1000, 1500];
sigmaFactors = [%40;
		80];
for sf = 1:length(sigmaFactors)
  keypointSuffix = sprintf('_%d_neigh_ROcontext_keypoint_sift.mat', sigmaFactors(sf));
   parfor k = 1:length(dictionarySize)
       params = struct('numTextonImages', 1000, 'dictionarySize', dictionarySize(k));
       CalculateDictionary(data.filenames,image_dir,data_dir,keypointSuffix, params, canSkip_dictionaryBuild);
       BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,keypointSuffix,params, canSkip_dictionaryBuild);
   end
end

%%
% Cluster Coarse Keypoint SIFT
%Pick 1000 images to do k_means on
canSkip_dictionaryBuild = 0;

clear params;
dictionarySize = [%20, 40, 50, 100, 
                200, 400, 800, 1000];
sigmaFactors = [%40;
		80];
for sf = 1:length(sigmaFactors)
  keypointSuffix = sprintf('_%d_neigh_ROcontext_keypoint_sift_context.mat', sigmaFactors(sf));
   parfor k = 1:length(dictionarySize)
       params = struct('numTextonImages', 1000, 'dictionarySize', dictionarySize(k));
       CalculateDictionary(data.filenames,image_dir,data_dir,keypointSuffix, params, canSkip_dictionaryBuild);
       BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,keypointSuffix,params, canSkip_dictionaryBuild);
   end
end

%% Build 2D Histograms

% Next up, we would construct 2D Histograms 
neighbourhoodSize = [ %40;
                       80;
                       ];
contextDictionarySize = [200, 400, 800, 1000];%[10 ,20, 40];
keypointDictionarySize = [200, 400, 800, 1000, 1500];
parfor j = 1: length(keypointDictionarySize)
     for k = 1: length(contextDictionarySize)
      for i = 1:length(neighbourhoodSize)
            %pfig = sp_progress_bar(sprintf('2D Hist of Neigh Size%d Keypt Dic%d Context Dic%d',neighbourhoodSize(i), keypointDictionarySize(j), contextDictionarySize(k)));
            CompileNeighborhoodPoolFeature2DSoftAssignHistogram( data.filenames,data_dir, sprintf('_soft_texton_ind_%d_%d_neigh_ROcontext_keypoint_sift',keypointDictionarySize(j),neighbourhoodSize(i)), sprintf('_soft_texton_ind_%d_%d_neigh_ROcontext_keypoint_sift_context',contextDictionarySize(j),neighbourhoodSize(i)), [keypointDictionarySize(j), contextDictionarySize(k)],[] , 0 );
            %close(pfig)
      end
    end
end

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
            
            inFName = fullfile(data_dir, sprintf('2Dhistograms_%s_with%s.mat', sprintf('_soft_texton_ind_%d_%d_neigh_ROcontext_keypoint_sift',config_table(i,3),config_table(i,2)), sprintf('_soft_texton_ind_%d_%d_neigh_ROcontext_keypoint_sift_context', config_table(i,1), config_table(i,2))));
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
            accuracy_results(k+1, (i-1)*length(keypointDictionarySize)+j+1) = acc_log(idx);
            idx= idx+1;
          end      
     end
end



outfname = fullfile( data_dir, 'results', '2DWeirdContextHist_accuracy.mat');
save(outfname, 'accuracy_results');
