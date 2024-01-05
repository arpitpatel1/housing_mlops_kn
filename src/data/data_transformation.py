import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from src.utils.exception import CustomException
import os
from pathlib import Path

from src.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    output_folder: Path = Path.cwd() / "models"
    preprocessor_obj_file_path: Path = output_folder / "preprocessor.pkl"


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation

        '''
        try:

            columns_to_encode = ['property_type', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category',
                                 'floor_category']
            numerical_columns = ['bedRoom', 'bathroom', 'built_up_area', 'pooja room', 'store room']

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'pooja room', 'store room']),
                    ('sector', OneHotEncoder(drop='first',handle_unknown='ignore'), ['sector']),
                    ('cat', OrdinalEncoder(), columns_to_encode)

                ],
                remainder='passthrough'
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_df, test_df):

        try:
            # train_df = pd.read_csv(train_path)
            # test_df = pd.read_csv(test_path)

            # logging.info("Read train and test data completed")
            #
            # logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = 'price'

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]


            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            target_feature_train_arr = target_feature_train_df.values.reshape(-1, 1)
            target_feature_test_arr = target_feature_test_df.values.reshape(-1, 1)
            input_feature_train_arr = input_feature_train_arr.toarray()
            input_feature_test_arr = input_feature_test_arr.toarray()


            # Combine the transformed input features with target features
            train_arr = np.hstack((input_feature_train_arr, target_feature_train_arr))
            test_arr = np.hstack((input_feature_test_arr, target_feature_test_arr))

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
