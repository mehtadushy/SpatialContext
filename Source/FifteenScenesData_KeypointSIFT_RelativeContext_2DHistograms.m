%Build 2D Histograms of keypoint+context features, such that appearance and
%context are captured along orthogonal axis
%Clustering appearance and context together didn't work, as expected -
%increasing the dictionary size to very large values (comparable to the 2D
%histogram size) does help in the joint clustering approach
clear;
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_RotImages3';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/newDir';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'rot3_scenes_names.mat'));

%Compute keypoint sift descriptors using vl_feat, and normalize the
%descriptors.
% Uncomment if Keypoint Descriptors have not been computed yet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% Flags 
DoG_Keypoints = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
canSkip_SIFTComputation = 0;
pfig = sp_progress_bar(sprintf('Computing Keypoint SIFT'));
if(DoG_Keypoints == true)
GenerateDoGKeypointSiftDescriptors( data.filenames,image_dir,data_dir,[],canSkip_SIFTComputation,pfig);
else
GenerateKeypointSiftDescriptors( data.filenames,image_dir,data_dir,[],canSkip_SIFTComputation,pfig);
end
close(pfig);

%Compute relative context of keypoint
params.sigmaFactors = [
                       80;
                       ];
params.keypointFeatureSuffix = '_keypoint_sift.mat';
ComputeRelativeKeypointContext(data.filenames, data_dir, params, canSkip_SIFTComputation);

%%
%It seems wasteful to go in an re-read the computed keypoint context features
% and put them in a separate file to be clustered separately, but meh!
neighbourhoodSize = [
                       80;
                       ];
for i = 1:length(neighbourhoodSize)
        CreateKeypointSIFTContextFeatures(data.filenames, data_dir, sprintf('_keypoint_sift_relative_context_neigh%d.mat', neighbourhoodSize(i)), params, 0);
end

%%
% Cluster Keypoint SIFT
%Pick 1000 images to do k_means on
params.numTextonImages = 1000;
dictionarySize = [50, 100, 200, 400, 800, 1000];
parfor k = 1:length(dictionarySize)
    pfig = sp_progress_bar(sprintf('Computing Keypoint SIFT Dictionaries With Dic Size %d',dictionarySize(k)));
    para = struct('numTextonImages', 1000, 'dictionarySize', dictionarySize(k));
    CalculateDictionary(data.filenames,image_dir,data_dir,'_keypoint_sift.mat',para, 0, pfig);
    BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,'_keypoint_sift.mat',para, 0, pfig);
    close(pfig)
end

%%
% Cluster Keypoint Context
%Pick 1000 images to do k_means on
params.numTextonImages = 1000;
dictionarySize = [10, 20 ,40];
parfor k = 1:length(dictionarySize)
    for i = 1:length(neighbourhoodSize)
    pfig =                     sp_progress_bar(sprintf('Computing Keypoint Context Dictionaries With Dic Size %d',dictionarySize(k)));
    para = struct('numTextonImages', 1000, 'dictionarySize', dictionarySize(k));
%    params.dictionarySize = dictionarySize(k);
    CalculateHistogramDictionary(data.filenames,data_dir,sprintf('_only__keypoint_sift_relative_context_neigh%d.mat', neighbourhoodSize(i)),para, 0, pfig);
    BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,sprintf('_only__keypoint_sift_relative_context_neigh%d.mat', neighbourhoodSize(i)),para, 0, pfig);
    close(pfig)
    end
end


%% Next up, we would construct 2D Histograms 
neighbourhoodSize = [
                       80;
                       ];
contextDictionarySize = [10 ,20, 40];
keypointDictionarySize = [50, 100, 200, 400,800, 1000];
parfor j = 1: length(keypointDictionarySize)
    for k = 1: length(contextDictionarySize)
      for i = 1:length(neighbourhoodSize)
            pfig = sp_progress_bar(sprintf('2D Hist of Neigh Size%d Keypt Dic%d Context Dic%d',neighbourhoodSize(i), keypointDictionarySize(j), contextDictionarySize(k)));
            CompileNeighborhoodPoolFeature2DSoftAssignHistogram( data.filenames,data_dir, sprintf('_soft_texton_ind_%d_keypoint_sift',keypointDictionarySize(j)), sprintf('_soft_texton_ind_%d_only__keypoint_sift_relative_context_neigh%d', contextDictionarySize(k), neighbourhoodSize(i)), [keypointDictionarySize(j), contextDictionarySize(k)],[] , 0, pfig );
            close(pfig)
      end
      
    end
end

%Using LibSVM Train

load(fullfile(data_dir2, 'rot3_category_names.mat'));
%Do a test-train split (data is already randomized
%train_percent = 90;
%train.index = (1 : floor(length(data.labels)*train_percent/100));
%train.labels = data.labels(train.index);
%test.index = (length(train.index)+1 : length(data.labels));
%test.labels = data.labels(test.index);
train_test_datafilename = fullfile(data_dir, 'train_test_data.mat');
save(train_test_datafilename, 'train', 'test');

%%
%%%%%%%%%%%%%%%%
% 2D Histograms of keypoint appearance captured with SIFT and keypoint
% context
%%%%%%%%%%%%%%%%


accuracy_results = cell(length(contextDictionarySize)+1, length(neighbourhoodSize)*length(keypointDictionarySize)+1);

for k = 1: length(contextDictionarySize)
     accuracy_results{k+1, 1} = sprintf('%d_Context', contextDictionarySize(k));
     for i = 1:length(neighbourhoodSize)
          for j = 1: length(keypointDictionarySize)
            accuracy_results{1, (i-1)*length(keypointDictionarySize)+j+1} = sprintf('%dN %d_KD', neighbourhoodSize(i), keypointDictionarySize(j));
          end      
     end
end

%%
n_sz = length(neighbourhoodSize);
k_sz = length(keypointDictionarySize);
c_sz = length(contextDictionarySize);

for k = 1: length(contextDictionarySize)
     for i = 1:length(neighbourhoodSize)
         acc_log = cell(1, k_sz);
         parfor j = 1: length(keypointDictionarySize)
            %fprintf('2D Hist of Neigh Size%d Keypt Dic%d Context Dic%d',neighbourhoodSize(i), keypointDictionarySize(j), contextDictionarySize(k));
            
            inFName = fullfile(data_dir, sprintf('2Dhistograms_%s_with%s.mat', sprintf('_soft_texton_ind_%d_keypoint_sift',keypointDictionarySize(j)), sprintf('_soft_texton_ind_%d_only__keypoint_sift_relative_context_neigh%d', contextDictionarySize(k), neighbourhoodSize(i))));
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
                
            bestcv = 0;
            for log2c = -2:2:4
                params = sprintf('-q -c %d -v 2 -t 4', 2^log2c);
                cv = svmtrain(train.labels',train.K, params );
                if (cv >= bestcv),
                  bestcv = cv; bestc = 2^log2c; bestlog2c = log2c;
                end
            end
            fprintf('Best C = %d\n',bestc);
            bestcv = 0;
            for log2c = (bestlog2c-0.3):0.1:(bestlog2c+0.3)
                params = sprintf('-q -c %d -v 2 -t 4', 2^log2c);
                cv = svmtrain(train.labels',train.K, params );
                if (cv >= bestcv),
                  bestcv = cv; C = 2^log2c;
                end
            end
            fprintf('C = %d\n',bestc);
            params = sprintf('-q -c %d -t 4', C);
            bow_model = svmtrain(train.labels',train.K, params );
            [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
            acc_log{j} = accuracy_P(1);
         end    
         accuracy_results(k+1 ,(i-1)*k_sz+2:i*k_sz+1) = acc_log;

     end
end


outfname = fullfile( data_dir, 'results', '2DContextHist_accuracy.mat');
save(outfname, 'accuracy_results');














