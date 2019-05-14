%Build spatial pyramids for the scenes categories dataset
%%Run the file cell wise
clear;
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/SoftAssign_2D';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'scenes_names.mat'));

%Compute sift descriptors with a grid spacing of 8 and patch size of 16
pfig = sp_progress_bar(sprintf('Computing Dense SIFT'));
GenerateSiftDescriptors( data.filenames,image_dir,data_dir,[],0,pfig);
close(pfig);
%Compute keypoint sift descriptors using vl_feat, and normalize the
%descriptors

pfig = sp_progress_bar(sprintf('Computing Keypoint SIFT'));
GenerateKeypointSiftDescriptors( data.filenames,image_dir,data_dir,[],0,pfig);
close(pfig);



%Parameters to skip the building of various stuff, if it already exists in
%the data directory.
%IF you want the stuff recomputed, set the corresponding flag to false
skip_rebuild_dictionaries = 1;
skip_pyramid_rebuild = 1;
skip_rebuild_textons = skip_rebuild_dictionaries; %Change when dictionories are changed
skip_compute_interest_points = 1;


%Pick 1000 images to do k_means on
params.numTextonImages = 1000;


%%
%Pick 1000 images to do k_means on
params.numTextonImages = 1000;
% First cluster the dense sifts and create textons for those
dictionarySize = [50, 100, 200, 400];
for k = 1:length(dictionarySize)
    pfig = sp_progress_bar(sprintf('Computing Dense SIFT Dictionaries With Dic Size %d',dictionarySize(k)));
    params.dictionarySize = dictionarySize(k);
    CalculateDictionary(data.filenames,image_dir,data_dir,'_sift.mat',params, 0, pfig);
    BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,'_sift.mat',params, 0, pfig);
    close(pfig)
end

%%
% Build Features for Keypoint Neighbourhood Pooling
%Pick 1000 images to do k_means on
params.numTextonImages = 1000;
% First cluster the keypoint sifts and create textons for those
dictionarySize = [50, 100, 200, 400];
for k = 1:length(dictionarySize)
    pfig = sp_progress_bar(sprintf('Computing Keypoint SIFT Dictionaries With Dic Size %d',dictionarySize(k)));
    params.dictionarySize = dictionarySize(k);
    CalculateDictionary(data.filenames,image_dir,data_dir,'_keypoint_sift.mat',params, 0, pfig);
    BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,'_keypoint_sift.mat',params, 0, pfig);
    close(pfig)
end

%% Now we compile dense descritptors in the neighborhood of keypoints and
%store them in the data directory
params.sigmaFactors = [6 10;                %%%%This is Radius not Dia%%%%
                       %6 14;
                       %8 16;
                       10 16;
                       %10 18
                       ];
%This is used to fetch the scale and locations of SIFT Keypoints
params.keypointFeatureSuffix = '_keypoint_sift.mat';
dictionarySize = [50, 100, 200, 400];
for k = 1:length(dictionarySize) %This is really an iteration through dictionary sizes of dense sift. 
                                 %The dictionary size of keypoint sift does
                                 %not come into play here
    params.dictionarySize = dictionarySize(k);
    fprintf('Computing Keypoint Neighbourhood dense pooling for dense dictionary size %d\n', params.dictionarySize);
    ComputeNeighborhoodPoolingSoftAssignHistogram( data.filenames,data_dir, sprintf('_soft_texton_ind_%d_sift.mat',params.dictionarySize), params, 0);
end
%% Next up create dictionaries for these neighbourhood histograms and assign
%labels to them
sigmaFactors = [6 10;
              % 6 14;
               %8 16;
               10 16;
               %10 18
               ];
neighborhoodHistogramDictionarySize = [50, 100, 200, 400];
dictionarySize = [20, 50, 80, 100, 120];
params.numTextonImages = 200;
for k = 1: length(neighborhoodHistogramDictionarySize)
    for sf = 1:size(sigmaFactors,1)
        for dic = 1:length(dictionarySize)
            pfig = sp_progress_bar(sprintf('Computing With SF %d_%d HistDic %d Dic Size %d',sigmaFactors(sf,1), sigmaFactors(sf,2), neighborhoodHistogramDictionarySize(k),dictionarySize(dic)));
            params.dictionarySize = dictionarySize(dic);
            CalculateHistogramDictionary( data.filenames,data_dir, sprintf('_%d_%d_neighborhood_histsize%d.mat', sigmaFactors(sf,1), sigmaFactors(sf,2), neighborhoodHistogramDictionarySize(k))  , params, 0, pfig );
            SoftAssignHistogramClusters( data.filenames,data_dir, sprintf('_%d_%d_neighborhood_histsize%d.mat', sigmaFactors(sf,1), sigmaFactors(sf,2), neighborhoodHistogramDictionarySize(k)) , params, 0, pfig );
            close(pfig)
        end
    end
end
    

%% Next up, we would construct 2D Histograms 
sigmaFactors = [6 10;
               %6 14;
               %8 16;
               10 16;
               %10 18
               ];
neighborhoodHistogramDictionarySize = [50, 100, 200, 400];
dictionarySize = [20, 50, 80, 100, 120]; %%%This is the size of the histogram of histograms
keypointDictionarySize = [50, 100, 200, 400];
for j = 1: length(keypointDictionarySize)
    for k = 1: length(neighborhoodHistogramDictionarySize)
      for sf = 1:size(sigmaFactors,1)
        for dic = 1:length(dictionarySize)
            pfig = sp_progress_bar(sprintf('Computing With SF %d_%d HistDic %d Dic Size %d',sigmaFactors(sf,1), sigmaFactors(sf,2), neighborhoodHistogramDictionarySize(k),dictionarySize(dic)));
            CompileNeighborhoodPoolFeature2DSoftAssignHistogram( data.filenames,data_dir, sprintf('_soft_texton_ind_%d_keypoint_sift',keypointDictionarySize(j)), sprintf('_soft_texton_ind_%d_%d_%d_neighborhood_histsize%d', dictionarySize(dic), sigmaFactors(sf,1), sigmaFactors(sf,2), neighborhoodHistogramDictionarySize(k)), [keypointDictionarySize(j), dictionarySize(dic)], params, 0, pfig );
            close(pfig)
        end
      end
    end
end
%%


