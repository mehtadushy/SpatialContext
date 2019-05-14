%Build 2D Histograms of keypoint+context features, such that appearance and
%context are captured along orthogonal axis
%Features get context pooled independently in 4 sectors, resulting in 4
%different set of features, which can be used to construct 4 histograms
clear;
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/SoftAssign_2D';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'scenes_names.mat'));

%Compute keypoint sift descriptors using vl_feat, and normalize the
%descriptors.
% Uncomment if Keypoint Descriptors have not been computed yet
%pfig = sp_progress_bar(sprintf('Computing Keypoint SIFT'));
%GenerateKeypointSiftDescriptors( data.filenames,image_dir,data_dir,[],0,pfig);
%close(pfig);

%Compute relative context of keypoint
params.sigmaFactors = [40;
                       80;
                       ];
params.keypointFeatureSuffix = '_keypoint_sift.mat';
%The following would create _directional_keypoint_sift.mat (which is just
%keypoint_sift with each entry duplicated 4x) and
%_directional_relative_context_neigh%d.mat

ComputeDirectionalRelativeKeypointContext(data.filenames, data_dir, params, 0);
%RemoveCountFromDirContext(data.filenames, data_dir, params, 0);
%%
%It seems wasteful to go in an re-read the computed keypoint context features
% and put them in a separate file to be clustered separately, but meh!

% for i = 1:length(neighbourhoodSize)
%         CreateKeypointSIFTContextFeatures(data.filenames, data_dir, sprintf('_keypoint_sift_directional_relative_context_neigh%d.mat', neighbourhoodSize(i)), params, 0);
% end

%%
% Cluster Keypoint SIFT
%Pick 1000 images to do k_means on
params.numTextonImages = 1000;
dictionarySize = [50, 100, 200, 400];
canSkip_dictionaryBuild = 1;
for k = 1:length(dictionarySize)
    pfig = sp_progress_bar(sprintf('Computing Keypoint SIFT Dictionaries With Dic Size %d',dictionarySize(k)));
    params.dictionarySize = dictionarySize(k);
    CalculateDictionary(data.filenames,image_dir,data_dir,'_keypoint_sift.mat',params, canSkip_dictionaryBuild, pfig);
    %Rename dictionary to duplicate keypt Sift dictionary and map
    inFName = fullfile(data_dir, sprintf('dictionary_%d_keypoint_sift.mat', params.dictionarySize));
    load(inFName, 'dictionary');
    outFName = fullfile(data_dir, sprintf('dictionary_%d_directional_keypoint_sift.mat', params.dictionarySize));
    save(outFName, 'dictionary');
    BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,'_directional_keypoint_sift.mat',params, 0, pfig);
    close(pfig)
end

%%
% Cluster Keypoint Context
%Pick 1000 images to do k_means on
params.numTextonImages = 250;
 neighbourhoodSize = [  40;
                        80;
                        ];
dictionarySize = [10, 20, 40];
for k = 1:length(dictionarySize)
    for i = 1:length(neighbourhoodSize)
    pfig = sp_progress_bar(sprintf('Computing Keypoint Context Dictionaries With Dic Size %d',dictionarySize(k)));

    params.dictionarySize = dictionarySize(k);
    CalculateHistogramDictionary(data.filenames,data_dir,sprintf('_directional_relative_context_neigh%d.mat', neighbourhoodSize(i)),params, 0, pfig);
    BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,sprintf('_directional_relative_context_neigh%d.mat', neighbourhoodSize(i)),params, 0, pfig);
    close(pfig)
    end
end


% Next up, we would construct 2D Histograms 
neighbourhoodSize = [  40;
                       80;
                       ];
contextDictionarySize = [10, 20, 40];
keypointDictionarySize = [50, 100, 200, 400];
parfor j = 1: length(keypointDictionarySize)
    for k = 1: length(contextDictionarySize)
      for i = 1:length(neighbourhoodSize)
            pfig = sp_progress_bar(sprintf('2D Hist of Neigh Size%d Keypt Dic%d Context Dic%d',neighbourhoodSize(i), keypointDictionarySize(j), contextDictionarySize(k)));
            CompileNeighborhoodPoolFeature2DSoftAssignHistogram( data.filenames,data_dir, sprintf('_soft_texton_ind_%d_directional_keypoint_sift',keypointDictionarySize(j)), sprintf('_soft_texton_ind_%d_directional_relative_context_neigh%d', contextDictionarySize(k), neighbourhoodSize(i)), [keypointDictionarySize(j), contextDictionarySize(k)], params, 0, pfig );
            close(pfig)
      end
      
    end
end

%%

