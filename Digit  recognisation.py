{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.datasets import mnist "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = mnist.load_data('mnist.db')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "#dataset\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "train , test = dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train , y_train = train "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_test , y_test = test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 28, 28)"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000,)"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 28, 28)"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000,)"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train_1d = x_train.reshape(-1,28*28)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 784)"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train_1d.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_test_1d = x_test.reshape(-1,28*28)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 784)"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_test_1d.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train = x_train_1d.astype('float64')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 784)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_test = x_test_1d.astype('float64')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 784)"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "# here we used 0-9 output so we used categorical \n",
    "from keras.utils.np_utils import to_categorical"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train_cat = to_categorical(y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 10)"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train_cat.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import Sequential"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.layers import Dense"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Dense(units=512, input_dim=28*28, activation='relu'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "'Sequential' object has no attribute 'Summary'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-58-f405ab720750>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mSummary\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mAttributeError\u001b[0m: 'Sequential' object has no attribute 'Summary'"
     ]
    }
   ],
   "source": [
    "model.Summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Dense(units=128, activation='relu'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Dense(units=32, activation='relu'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Dense(units=10, activation='softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_2\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense_2 (Dense)              (None, 512)               401920    \n",
      "_________________________________________________________________\n",
      "dense_3 (Dense)              (None, 128)               65664     \n",
      "_________________________________________________________________\n",
      "dense_4 (Dense)              (None, 32)                4128      \n",
      "_________________________________________________________________\n",
      "dense_5 (Dense)              (None, 10)                330       \n",
      "=================================================================\n",
      "Total params: 472,042\n",
      "Trainable params: 472,042\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.optimizers import Adam"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer=Adam(learning_rate=0.00001),loss='binary_crossentropy', metrics=['accuracy'])\n",
    "                  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n",
      "60000/60000 [==============================] - 30s 502us/step - loss: 1.2820 - accuracy: 0.8920\n",
      "Epoch 2/20\n",
      "60000/60000 [==============================] - 30s 498us/step - loss: 0.6566 - accuracy: 0.9295\n",
      "Epoch 3/20\n",
      "60000/60000 [==============================] - 30s 502us/step - loss: 0.2897 - accuracy: 0.9348\n",
      "Epoch 4/20\n",
      "60000/60000 [==============================] - 31s 517us/step - loss: 0.1693 - accuracy: 0.9520\n",
      "Epoch 5/20\n",
      "60000/60000 [==============================] - 33s 552us/step - loss: 0.1237 - accuracy: 0.9654\n",
      "Epoch 6/20\n",
      "60000/60000 [==============================] - 32s 529us/step - loss: 0.0964 - accuracy: 0.9736\n",
      "Epoch 7/20\n",
      "60000/60000 [==============================] - 30s 500us/step - loss: 0.0793 - accuracy: 0.9783\n",
      "Epoch 8/20\n",
      "60000/60000 [==============================] - 30s 502us/step - loss: 0.0675 - accuracy: 0.9814s - los\n",
      "Epoch 9/20\n",
      "60000/60000 [==============================] - 31s 521us/step - loss: 0.0583 - accuracy: 0.9837\n",
      "Epoch 10/20\n",
      "60000/60000 [==============================] - 32s 528us/step - loss: 0.0514 - accuracy: 0.9857\n",
      "Epoch 11/20\n",
      "60000/60000 [==============================] - 35s 587us/step - loss: 0.0459 - accuracy: 0.9870s - loss: 0.0458 - \n",
      "Epoch 12/20\n",
      "60000/60000 [==============================] - 39s 647us/step - loss: 0.0412 - accuracy: 0.9885\n",
      "Epoch 13/20\n",
      "60000/60000 [==============================] - 37s 612us/step - loss: 0.0372 - accuracy: 0.9894\n",
      "Epoch 14/20\n",
      "60000/60000 [==============================] - 32s 537us/step - loss: 0.0337 - accuracy: 0.9904\n",
      "Epoch 15/20\n",
      "60000/60000 [==============================] - 35s 575us/step - loss: 0.0308 - accuracy: 0.9912s - loss: 0.0308 - accuracy: 0.99\n",
      "Epoch 16/20\n",
      "60000/60000 [==============================] - 33s 547us/step - loss: 0.0283 - accuracy: 0.9919\n",
      "Epoch 17/20\n",
      "60000/60000 [==============================] - 32s 537us/step - loss: 0.0261 - accuracy: 0.9925\n",
      "Epoch 18/20\n",
      "60000/60000 [==============================] - 32s 535us/step - loss: 0.0241 - accuracy: 0.9930\n",
      "Epoch 19/20\n",
      "60000/60000 [==============================] - 36s 593us/step - loss: 0.0225 - accuracy: 0.9935\n",
      "Epoch 20/20\n",
      "60000/60000 [==============================] - 32s 527us/step - loss: 0.0208 - accuracy: 0.9940\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.callbacks.History at 0x1f3d85ff288>"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x_train, y_train_cat, epochs=20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_img = x_test.reshape(-1,28*28)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 784)"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_img.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_test_1d = model.predict(test_img)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([9.3865385e-23, 1.0000000e+00, 1.9010627e-17, 4.0790150e-18,\n",
       "       4.8385453e-23, 0.0000000e+00, 1.6162237e-20, 9.5416876e-19,\n",
       "       0.0000000e+00, 2.8657060e-22], dtype=float32)"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test_1d[5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x1f3f96ebfc8>"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAADEAAAD8CAYAAADe6kx2AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAHX0lEQVR4nO2dXYxVVxXHf//OLcNHoczgR5A2AVKqgWRikTSgPhhra0uM+NAH+iJqTROrSfXFQHzyscZYY6K2jR9pjJG2SCqSGmq0PopCrGMRBoZi21EsWLE2WIePLh/OGnKmzp27h7n3su6wfsnN3Wefxb7nx7n7zJ3Z664jM6PXueZKH0A7SIkopEQUUuJykHSnpBFJo5K2t2VQM+vaA+gDjgOrgXnAH4G1sx2322fiVmDUzF4ws3PATmDLbAfttsQK4OXa9pj3XULSfZIOSDrQUMMknW41aLclNEXfpM89ZvaomW0wsw0LWQLwYqtBuy0xBtxY274B+FuzYDUaRYN2W+L3wBpJqyTNA7YCe5oF24ULRYOWqbYJM7sg6QvAPqor1Q/M7NBsx+2qBICZPQ083c4x8yd2FFKi42iqHyv/T2yJwt//Y0sUkhJRSIlOc/PQf4riQkscHV5YFBdaopSUiEJKRCG0RF5ie43QEmuGzhbFhZY4NryoKC60RCkpEYWU6DS6puzwQkvYm28WxYWWKCUlopASUUiJKFwdEpJulPSspMOSDkl6wPsHJf1S0jF/HvB+SfqW524MS1pfG2ubxx+TtK1tFgX5GMuB9d5eDBwF1gJfA7Z7/3bgQW9vBn5BtfC+Edjv/YPAC/484O2B6V57MQMGHGh5jJeRZPIz4HZgBFheEx3x9iPAPbX4Ed9/D/BIrX9S3GwkZjQnJK0EbgH2A+80s5N+Nk8C7/CwZvkbLfM6/DUu5XacZ7zouIolJF0H/BT4opn9e7rQKfpsmv7JHbXcjmvpLzq2IglJ11IJ/NjMdnv3K5KW+/7lwCnvb5a/MaO8jplQcnUS8H3gsJl9o7ZrDzBxhdlGNVcm+j/pV6mNwGv+dtsH3CFpwK9kd3jf7CmYyB+kOu3DwHP+2AwsA34FHPPnQY8X8G2qDLM/ARtqY30GGPXHp1u99op1S4omtqxwmfVKsESD9jpnDprZhuniro6f2L1ASnQa9c8rigstYePniuJCS4yvnAPrE/1/mQMrRaWkRBRCS+Tqaa+RElFIiSikRBRSIgopEYWUiEJKRCElopASUUiJKFxdEpL6JP1B0l7fXiVpv6c4PO5fEkdSv2+P+v6VtTF2eP+IpI92XQJ4ADhc234QeMjM1gBngHu9/17gjJndBDzkcUhaS/VN+HXAncB3JPXN7vCdwlSIG6hWSD8M7KVaIf0H0PD9m4B93t4HbPJ2w+ME7AB21Ma8FNettIhvAl8GJnI8lwH/MrOJIgL1FIdL6Q++/zWPv3JpEZI+Bpwys4P17ilCrcW+GadFvGfofKvDA8pqFHwA+LikzcB8YAnVmVkqqeH/2/UUh4n0hzFJDeB64J9cRlrE8eHrqKZbC2aYJvQhYK+3nwS2evth4H5vfx542NtbgSe8vY6qdk0/sIoq36mvHXNiNhKrgd9RpTg8CfR7/3zfHvX9q2v//itU6RIjwF2tXq9UItMiopASUUiJTpMLj71GSkQhJaKQElFIiSikRBRSIgqhJebER/FSQkvk7xO9RmiJOTGxc070GikRhZToNHmJjUKeibciaamkXZKOeOmLTT1V8sIXJh8DPuvtecBSeqnkBdXi+wmoVlpr/T1V8mI1cBr4oacKfU/SInqs5EUDWA9818xuAc5SvX2a0bbcjnVDFwsOr0xiDBgzs/2+vYtKquMlL9p2dTKzvwMvS3q3d90G/JleKnnhk/C9wAGqshdPUV1dOl7y4n1D/Znb0VOkRBRSIgopEYWUiEJKRCElopASUUiJKKREFEJL5F/Fe42UiEJKdJq8OvUaKRGF0BIXl82BG5D1vToHbgVXSmyJRQuKwkrTIr7kdwF5XtJPJM3vSsmLs28USZQsOq6gyihY4NtPAJ/y5/o34z/n7fuZ/M34x729lsnfjD9Om74ZX/p2agALvObAQuAkVfmLXb7/MeAT3t7i2/j+2/zOCVuAnWY2bmYnqFZQb53uRUs/drQstGBmf5X0deAl4A3gGeAghSUvJNVLXvy2NnTTtAjgPoD5tOljhy+cb6F6C7wLWATcNUVo20tetPNOIB8BTpjZaTM7D+wG3o+XvPCYqUpeMNuSF6WUSLwEbJS00N/bE2kRzwJ3e8xb0yIm0iXuBn5t1czeA2z1q9cqYA1VNYmmlM6J0rSIrwJHgOeBH1FdYbLkRQlzIi0ib93Qa6REp8m/dvQaKRGFlIhCSkQhJaKQEp0mPzv1GikRhZSIQkpEISWikBJRSIkopESnyd8neo2UiMKckAi9BCzpv8DrZvb26eKin4mLrQQgvkQRKdEFdpcEhZ7YpUQ/E0WkRCeR9IqkcX+MTRcbUsLvrjZIlaa3GHjV78I2JSElqFJNLwAvmtk5YCdVMuWUlNy760qwgkriGUkGPEeVzjolUc+EgJ+b2Xqq9NXbqeqmTUlUiTGqhGDM7BRwFGia/R5V4hBwsyfVL6W6c+FTzYKjzom3UWUvH6F6a/3GzB5tFpwfO6KQElFIiSikRBT+B/atxokMYL8BAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(y_test_1d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
