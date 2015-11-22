addpath /data/vision/torralba/regionmem/memorability_cnn/lib/lmdbLib;
addpath /data/vision/torralba/regionmem/memorability_cnn/lib/matlab-lmdb;
addpath /data/vision/torralba/regionmem/mexopencv;
addpath /data/vision/scratch/torralba/khosla/itracker/deep_model/caffe-cudnn3/build/src/caffe/proto_matlab;

if(matlabpool('size')==0), try, matlabpool; end; end

image_set = 'test';
dataset_folder = ['cifar10_lmdb_matlab/' image_set];
output_folder = fileparts(dataset_folder);
if(~isempty(output_folder) && ~exist(output_folder, 'dir'))
  mkdir(output_folder);
end

images_folder = '/data/vision/scratch/torralba/khosla/itracker/deep_model/caffe-cudnn3/data/cifar10/images';
[image_list, labels] = textread(['/data/vision/scratch/torralba/khosla/itracker/deep_model/caffe-cudnn3/data/cifar10/' image_set '.txt'], '%s %d');
image_list = cellfun(@(x) [images_folder '/' x], image_list, 'UniformOutput', false);

c = lmdbConfig();

fprintf('[creating dataset: %s]\n', dataset_folder);
database = lmdb.DB(dataset_folder, 'MAPSIZE', c.mapsize);

if(c.randomize)
  rp = randperm(length(image_list));
  image_list = image_list(rp);
  labels = labels(rp);
end

transaction = database.begin();

batchSize = 1000;
numBatches = ceil(length(image_list)/batchSize);
numChannels = 3;
numWidth = 32;
numHeight = 32;
dataField = 'data';
encoded = 1;

tic;
for i=1:numBatches
  startIdx = (i-1)*batchSize + 1;
  endIdx = min(i*batchSize, length(labels));

  batchLabels = labels(startIdx:endIdx);
  batchList = image_list(startIdx:endIdx);

  datums = cell(length(batchList), 1);
  parfor j=1:length(datums)
    datums{j} = pb_read_caffe__Datum();
    datums{j} = pblib_set(datums{j}, 'channels', numChannels);
    datums{j} = pblib_set(datums{j}, 'width', numWidth);
    datums{j} = pblib_set(datums{j}, 'height', numHeight);

    if(encoded)
      im = cv.imread(batchList{j}, 'Flags', 1);
      im = cv.imencode('.jpg', im, 'JpegQuality', 90);
      datums{j} = pblib_set(datums{j}, 'encoded', true);
    else
      im = imread(batchList{j});
      im = permute(im, [2 1 3]);
      im = im(:, :, [3 2 1]);
      datums{j} = pblib_set(datums{j}, 'encoded', false);
    end

    datums{j} = pblib_set(datums{j}, dataField, im(:)');
    datums{j} = pblib_set(datums{j}, 'label', batchLabels(j));
    datums{j} = pblib_generic_serialize_to_string(datums{j});
  end

  for j=1:length(datums)
    transaction.put(sprintf('%08d', startIdx+j-1), datums{j});
  end

  fprintf('[commiting transaction %d of %d, time: %f]\r', i, numBatches, toc);
  transaction.commit();
  if(i~=numBatches)
    transaction = database.begin();
  end
end
fprintf('\n');

