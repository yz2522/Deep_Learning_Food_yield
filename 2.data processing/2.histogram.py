import numpy as np
import scipy.io as io
import math

class fetch_data():
    def __init__(self):
        self.dir = "/Volumes/Elements/data_all/img_output/"
        # load yield data
        self.data_yield = np.genfromtxt('regions_yield.csv', delimiter=',')
        self.locations = np.genfromtxt('regions.csv', delimiter=',')
        # self.data_yield[0,0] = 2015.0

        # generate index for all data
        length = self.data_yield.shape[0]
        self.index_all = np.arange(length)

        self.year = 2012
        # divide data by year
        self.index_train=[]
        self.index_val=[]
        for i in range(self.data_yield.shape[0]):
            if self.data_yield[i,0]==self.year:
                self.index_val.append(i)
            else:
                self.index_train.append(i)
        self.index_train=np.array(self.index_train)
        self.index_val=np.array(self.index_val)

    def calc_mean(self):
        # traverse training data
        image_mean = np.zeros([32, 32, 9], dtype=np.float32)

        print('traversing training data to calc image_mean:')
        n = 0
        for i in self.index_train:
            year = str(int(self.data_yield[i, 0]))
            loc1 = str(int(self.data_yield[i, 1]))
            loc2 = str(int(self.data_yield[i, 2]))

            filename = year + '_' + loc1 + '_' + loc2 + '.mat'
            content = io.loadmat(self.dir + filename)
            image_temp = content['image_divide']
            image_temp = self.filter_size(image_temp, 48)
            image_temp = self.filter_timespan(image_temp, 49, 305, 7)
            image_temp = self.filter_abnormal(image_temp, 0, 5000)
            image_temp = self.filter_crop_mask(image_temp, year, loc1, loc2)


            ratio = float(n) / float(n + 1)
            image_mean *= ratio
            ratio = float(n + 1)
            image_temp /= ratio
            image_mean += image_temp
            n += 1

        # print image_mean
        image_mean = image_mean.astype(dtype='float32', copy=False)
        np.save('image_mean_48by48.npy', image_mean)
        return image_mean

    def filter_size(self,image_temp,size):
        return image_temp[0:size, 0:size, :].astype(np.float32)

    def filter_abnormal(self,image_temp,min,max):
        image_temp[image_temp<min]=min
        image_temp[image_temp>max]=max
        return image_temp

    def filter_timespan(self,image_temp,start_day,end_day,bands):
        start_index=int(math.floor(start_day/8))*bands
        end_index=int(math.floor(end_day/8))*bands
        if end_index>image_temp.shape[2]:
            image_temp = np.concatenate((image_temp, 
                np.zeros((image_temp.shape[0],image_temp.shape[1],end_index-image_temp.shape[2]))),axis=2)
        return image_temp[:,:,start_index:end_index]


    def calc_histogram(self,image_temp,bin_seq,bins,times,bands):
        hist=np.zeros([bins,times,bands])
        for i in range(image_temp.shape[2]):
            density, _ = np.histogram(image_temp[:, :, i], bin_seq,range = (np.nanmin(image_temp[:, :, i]), np.nanmax(image_temp[:, :, i])), density=False)
            # if density.sum()==0:
            #     continue
            hist[:, i / bands, i % bands] = density / float(density.sum())
        print(image_temp.shape[2])
        return hist

    def calc_histogram_flat(self,image_temp,bin_seq,bins,times,bands):
        hist=np.zeros([bins,times,bands])
        for i in range(image_temp.shape[1]):
            density, _ = np.histogram(image_temp[:, i], bin_seq, density=False)
            hist[:, i / bands, i % bands] = density / float(density.sum())
        return hist

    def quality_dector(self,image_temp):
        filter_0=image_temp>0
        filter_5000=image_temp<5000
        filter=filter_0*filter_5000
        return float(np.count_nonzero(filter))/image_temp.size

    def next_batch_hist(self,batch_size,type='train'):
        if type == 'train':
            index = self.index_train
        elif type == 'validate':
            index = self.index_val
        elif type == 'test':
            index = self.index_test
        index_batch = np.random.choice(index, batch_size)
        output_image = np.zeros([batch_size, 32, 32, 9])
        output_yield = np.zeros([batch_size])
        for n, i in enumerate(index_batch):
            year = str(int(self.data_yield[i, 0]))
            loc1 = str(int(self.data_yield[i, 1]))
            loc2 = str(int(self.data_yield[i, 2]))

            # filename = year + '_' + loc1 + '_' + loc2 + '.mat'
            filename = year + '_' + loc1 + '_' + loc2 + '.npz'
            content = np.load(self.dir + filename)
            image_temp = content['arr_0']
            image_temp = self.filter_timespan(image_temp, 49, 305, 9)

            bin_seq=np.linspace(1,4999,33)
            image_temp = self.calc_histogram(image_temp,bin_seq,32,32,9)
            image_temp[np.isnan(image_temp)] = 0
            if np.sum(image_temp) < 250:
                print('broken image',filename)
                print(np.isnan(image_temp))

            # output_image[n, :] = image_temp - self.image_mean
            output_image[n,:] = image_temp
            output_yield[n] = self.data_yield[i, 3]
            # print image_temp.shape
            # print np.sum(image_temp)
        return (np.float32(output_yield), np.float32(output_image))

    # save supervised data
    def save_data(self):
        output_image = np.zeros([self.index_all.shape[0], 32, 32, 9])
        output_yield = np.zeros([self.index_all.shape[0]])
        output_year = np.zeros([self.index_all.shape[0]])
        output_locations = np.zeros([self.index_all.shape[0],2])
        output_index = np.zeros([self.index_all.shape[0],2])

        for i in self.index_all:
            year = str(int(self.data_yield[i, 0]))
            loc1 = str(int(self.data_yield[i, 1]))
            loc2 = str(int(self.data_yield[i, 2]))

            key = np.array([int(loc1),int(loc2)])
            index = np.where(np.all(self.locations[:,0:2].astype('int') == key, axis=1))
            longitude = self.locations[index,2]
            latitude = self.locations[index,3]

            filename = year + '_' + loc1 + '_' + loc2 + '.npy'
            image_temp = np.load(self.dir + filename)
            image_temp = self.filter_timespan(image_temp, 49, 305, 9)
            print(image_temp.shape)

            bin_seq = np.linspace(1, 4999, 33)
            image_temp = self.calc_histogram(image_temp, bin_seq ,32, 32, 9)
            image_temp[np.isnan(image_temp)] = 0

            output_image[i, :] = image_temp
            output_yield[i] = self.data_yield[i, 3]
            output_year[i] = int(year)
            output_locations[i, 0] = longitude
            output_locations[i, 1] = latitude
            output_index[i,:] = np.array([int(loc1),int(loc2)])
            # print image_temp.shape
            print(i,np.sum(image_temp),year,loc1,loc2)
        np.savez(self.dir+'histogram_NY_PA_MD_full.npz',
                 output_image=output_image,output_yield=output_yield,
                 output_year=output_year,output_locations=output_locations,output_index=output_index)
        print('save done')

    # save mean data
    def save_data_mean(self):
        output_image = np.zeros([self.index_all.shape[0], 32*9])
        output_yield = np.zeros([self.index_all.shape[0]])
        output_year = np.zeros([self.index_all.shape[0]])
        output_locations = np.zeros([self.index_all.shape[0],2])
        output_index = np.zeros([self.index_all.shape[0],2])

        for i in self.index_all:
            year = str(int(self.data_yield[i, 0]))
            loc1 = str(int(self.data_yield[i, 1]))
            loc2 = str(int(self.data_yield[i, 2]))

            key = np.array([int(loc1),int(loc2)])
            index = np.where(np.all(self.locations[:,0:2].astype('int') == key, axis=1))
            longitude = self.locations[index,2]
            latitude = self.locations[index,3]

            filename = year + '_' + loc1 + '_' + loc2 + '.npy'
            image_temp = np.load(self.dir + filename)
            image_temp = self.filter_timespan(image_temp, 49, 305, 9)

            image_temp = np.sum(image_temp,axis=(0,1))/np.count_nonzero(image_temp)*image_temp.shape[2]
            image_temp[np.isnan(image_temp)] = 0

            output_image[i, :] = image_temp
            output_yield[i] = self.data_yield[i, 3]
            output_year[i] = int(year)
            output_locations[i, 0] = longitude
            output_locations[i, 1] = latitude
            output_index[i,:] = np.array([int(loc1),int(loc2)])
            print(i,np.sum(image_temp),year,loc1,loc2)
        np.savez(self.dir+'histogram_all_mean.npz',
                 output_image=output_image,output_yield=output_yield,
                 output_year=output_year,output_locations=output_locations,output_index=output_index)
        print('save done')

if __name__ == '__main__':
    data=fetch_data()
    data.save_data_mean()
    data.save_data()