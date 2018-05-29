round1_dir = 'OCT_data/train_balance_round1/';
res1_dir = 'OCT_test_traindata_balance_round1/images/';
round2_dir = 'OCT_data/train_balance_round2/';

imagefiles = dir([round1_dir '*.png']);
num_images = length(imagefiles);

for i = 1 : num_images
    imagename = [round1_dir imagefiles(i).name]
    image = imread(imagename);
    imagenum = strsplit(imagefiles(i).name, '.');
    resname = [res1_dir imagenum{1} '-outputs.png'];
    res = imread(resname);
    image(:, 1:256) = res;
    imwrite(image, [round2_dir imagefiles(i).name]);
end