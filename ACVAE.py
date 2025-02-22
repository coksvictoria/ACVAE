import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import os
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from ucimlrepo import fetch_ucirepo


"""DataTransformer module."""

from collections import namedtuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder
SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'
    ]
)


class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalize them to a scalar between [-1, 1]
    and a vector. Discrete columns are encoded using a OneHotEncoder.
    """

    def __init__(self, max_clusters=10, weight_threshold=0.005):
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

    def detect_constant_columns(self, df, threshold=0.9):
        """Detect columns with a large amount of one constant value.

        Args:
            df (pd.DataFrame): Dataframe to check.
            threshold (float): Proportion threshold for detection.

        Returns:
            list: List of columns with a large amount of one constant value.
        """
        constant_columns = []
        # Filter numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        for col in numerical_df.columns:
            most_frequent_value = numerical_df[col].mode()[0]
            proportion = (numerical_df[col] == most_frequent_value).mean()
            if proportion > threshold and (numerical_df[col] != most_frequent_value).any():
                constant_columns.append(col)
        return constant_columns

    def _fit_mixture(self, data):
        """Train MinMax Scaler for continuous columns.

        Args:
            data (pd.DataFrame): A dataframe containing a column.

        Returns:
            namedtuple: A ColumnTransformInfo object.
        """
        column_name = data.columns[0]
        non_zero_data = data[data[column_name] != 0]
        mn = MinMaxScaler(feature_range=(-1, 1))
        mn.fit(non_zero_data)

        return ColumnTransformInfo(
            column_name=column_name, column_type='mixture', transform=mn,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(2, 'softmax')],
            output_dimensions=3)

    def _fit_continuous(self, data):
        """Train Bayesian GMM for continuous columns.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        gm = ClusterBasedNormalizer(
            missing_value_generation='from_column',
            max_clusters=min(len(data), self._max_clusters),
            weight_threshold=self._weight_threshold
        )
        gm.fit(data, column_name)
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name, column_type='continuous', transform=gm,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components)

    def _fit_discrete(self, data):
        """Fit one hot encoder for discrete column.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, column_name)
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name, column_type='discrete', transform=ohe,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories)

    def fit(self, raw_data, discrete_columns=()):
        """Fit the ``DataTransformer``.

        Fits a ``ClusterBasedNormalizer`` for continuous columns and a
        ``OneHotEncoder`` for discrete columns.

        This step also counts the #columns in matrix data and span information.
        """
        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = True

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            # work around for RDT issue #328 Fitting with numerical column names fails
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        # Identify columns with float type
        float_columns = raw_data.select_dtypes(include=['float64']).columns

        # Check if all values in these float columns are actually integers
        int_like_float_columns = []

        for col in float_columns:
            if raw_data[col].apply(lambda x: x.is_integer()).all():
                int_like_float_columns.append(col)

        print("Columns with float type but containing only integer values:")
        print(int_like_float_columns)

        # Optional: Convert these columns to integer type
        for col in int_like_float_columns:
            raw_data[col] = raw_data[col].astype('int64')

        self._column_transform_info_list = []

        # Automatically detect mixture columns
        mixture_columns = self.detect_constant_columns(raw_data)

        print(mixture_columns)
        self._column_raw_dtypes = raw_data.infer_objects().dtypes

        for column_name in raw_data.columns:
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[[column_name]])
            elif column_name in mixture_columns:
                column_transform_info = self._fit_mixture(raw_data[[column_name]])
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]])

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]
        flattened_column = data[column_name].to_numpy().flatten()
        data = data.assign(**{column_name: flattened_column})
        gm = column_transform_info.transform
        transformed = gm.transform(data)

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same, but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
        index = transformed[f'{column_name}.component'].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output

    def _transform_mixture(self, column_transform_info, data, indicator_value=0):
        """Transform mixture of continuous and indicator values.

        Args:
            data (np.ndarray): Data to be transformed.
            indicator_value (numeric): Value indicating the presence of a special condition.

        Returns:
            tuple: Transformed data and scaler used for the transformation.
        """
        binary_indicator = np.where(data.values.flatten() == indicator_value, 1, 0)  # Accessing the underlying NumPy array
        df = pd.DataFrame({
            'binary_indicator': binary_indicator,
            'original_value': data.values.flatten()  # Accessing the underlying NumPy array
        })
        scaler = column_transform_info.transform
        df['scaled_value'] = scaler.transform(df['original_value'].values.reshape(-1, 1))

        # Update scaled_value to be -1 where binary_indicator is -1
        df.loc[df['binary_indicator'] == 1, 'scaled_value'] = -1

        output = np.zeros((len(df), 3))
        output[:, 0] = df['scaled_value']
        # Set the second column to 1 when binary_indicator is 0
        output[:, 1] = np.where(df['binary_indicator'] == 1, 1, 0)
        output[:, 2] = np.where(df['binary_indicator'] == 0, 1, 0)

        return output

    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()

    def _synchronous_transform(self, raw_data, column_transform_info_list):
        """Take a Pandas DataFrame and transform columns synchronously.

        Outputs a list with Numpy arrays.
        """
        column_data_list = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data))
            elif column_transform_info.column_type == 'discrete':
                column_data_list.append(self._transform_discrete(column_transform_info, data))
            elif column_transform_info.column_type == 'mixture':
                column_data_list.append(self._transform_mixture(column_transform_info, data))

        return column_data_list

    def _parallel_transform(self, raw_data, column_transform_info_list):
        """Take a Pandas DataFrame and transform columns in parallel.

        Outputs a list with Numpy arrays.
        """
        processes = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == 'continuous':
                process = delayed(self._transform_continuous)(column_transform_info, data)
            elif column_transform_info.column_type == 'discrete':
                process = delayed(self._transform_discrete)(column_transform_info, data)
            elif column_transform_info.column_type == 'mixture':
                process = delayed(self._transform_mixture)(column_transform_info, data)

            processes.append(process)
        results = Parallel(n_jobs=-1)(processes)
        return list(results)


    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        # Only use parallelization with larger data sizes.
        # Otherwise, the transformation will be slower.
        if raw_data.shape[0] < 500:
            column_data_list = self._synchronous_transform(
                raw_data,
                self._column_transform_info_list
            )
        else:

            column_data_list = self._parallel_transform(
                raw_data,
                self._column_transform_info_list
            )

        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes())).astype(float)
        data[data.columns[1]] = np.argmax(column_data[:, 1:], axis=1)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        return gm.reverse_transform(data)

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]

    def _inverse_transform_mixture(self, column_transform_info, column_data):
        """Inverse transform a mixture of continuous and indicator values.

        Args:
            column_transform_info (ColumnTransformInfo): Information about the column transformation.
            column_data (np.ndarray): Data to be transformed.

        Returns:
            np.ndarray: Inverse transformed data.
        """
        scaler = column_transform_info.transform

        data = pd.DataFrame(
            column_data[:, :2], columns=['value','flag']).astype(float)

        data[data.columns[1]] = np.argmax(column_data[:, 1:], axis=1)

        # Reverse scaling of continuous values
        data['value'] = scaler.inverse_transform(data['value'].values.reshape(-1, 1)).flatten()

        # Set original values to zero where flags are zero
        data.loc[data['flag'] == 0, 'value'] = 0

        return data['value']

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st)
            elif column_transform_info.column_type == 'discrete':
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data)
            elif column_transform_info.column_type == 'mixture':
                recovered_column_data = self._inverse_transform_mixture(
                    column_transform_info, column_data)

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes))
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data
		
		


def balance_classes(y_train):
    # Calculate class counts
    class_counts = np.bincount(y_train)

    # Determine the maximum count
    max_count = np.max(class_counts)

    # Calculate the number of samples needed for each class to balance
    samples_needed = max_count - class_counts

    # Create y_balanced with balanced classes
    y_bal_labels = []

    for label in range(len(class_counts)):
        y_bal_labels.extend([label] * samples_needed[label])

    y_bal_labels = np.array(y_bal_labels)

    return y_bal_labels

class CEncoderAux(nn.Module):
    def __init__(self, data_dim, compress_dims, embedding_dim, n_classes):
        super(CEncoderAux, self).__init__()
        dim = data_dim + n_classes
        seq = []
        for item in list(compress_dims):
            seq += [nn.Linear(dim, item), nn.ReLU()]
            dim = item

        self.seq = nn.Sequential(*seq)
        self.fc1 = nn.Linear(dim, embedding_dim)
        self.fc2 = nn.Linear(dim, embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x, c):
        """Encode the passed `input_`."""
        input_ = torch.cat([x, c], 1)
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        class_logits = self.classifier(mu)
        return mu,logvar, class_logits

class CEncoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(CEncoder, self).__init__()
        dim = data_dim + 2
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, x,c):
        """Encode the passed `input_`."""
        input_ = torch.cat([x, c], 1)
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar



class CDecoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(CDecoder, self).__init__()
        dim = embedding_dim+n_classes
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, z, c):
        """Decode the passed `input_`."""
        input_ = torch.cat([z, c], 1)
        return self.seq(input_), self.sigma

class ACVAE:
    def __init__(self, data_dim, compress_dims, decompress_dims, embedding_dim, n_classes, device, use_aux=False):
        self.device = device
        self.n_classes = n_classes
        self.use_aux = use_aux
        if use_aux:
            self.encoder = CEncoderAux(data_dim, compress_dims, embedding_dim, n_classes).to(device)
        else:
            self.encoder = CEncoder(data_dim, compress_dims, embedding_dim,n_classes).to(device)
        self.decoder = CDecoder(embedding_dim, decompress_dims, data_dim).to(device)
        self.opt = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.0001)
        self.best_loss = np.inf

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _loss_function(self, recon_x, x, sigmas, mu, logvar, output_info, factor):
        st = 0
        loss = []
        eps = 1e-6  # Small epsilon for numerical stability
        for column_info in output_info:
            for span_info in column_info:
                if span_info.activation_fn != 'softmax':
                    ed = st + span_info.dim
                    std = torch.clamp(sigmas[st], min=eps)  # Clamping to avoid log(0)
                    eq = x[:, st] - torch.tanh(recon_x[:, st])
                    loss.append((eq ** 2 / (2 * (std ** 2))).sum())
                    loss.append(torch.log(std + eps) * x.size(0))
                    st = ed
                else:
                    ed = st + span_info.dim
                    loss.append(F.cross_entropy(recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                    st = ed

        assert st == recon_x.size(1)
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return sum(loss) * factor / x.size(0) + KLD / x.size(0)

    def train(self, loader, epochs, early_stop_thresh, transformer, factor=1.0,save_interval=None):

        pbar = tqdm(total=epochs, desc="Training", position=0, leave=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for bat_idx, batch in enumerate(loader):
                real_x, real_y = batch
                real_x = real_x.to(self.device).float()
                real_y = F.one_hot(real_y, num_classes=self.n_classes).float().to(self.device)
                self.opt.zero_grad()
                if self.use_aux:
                    mu, logvar, class_logits = self.encoder(real_x, real_y)
                else:
                    mu, logvar = self.encoder(real_x, real_y)
                z = self.reparameterize(mu, logvar)
                recon_x, sigmas = self.decoder(z, real_y)
                loss_val = self._loss_function(recon_x, real_x, sigmas, mu, logvar, transformer.output_info_list, factor)

                if self.use_aux:
                    class_loss = F.cross_entropy(class_logits, real_y.argmax(dim=1))
                    loss_val += class_loss

                # Compute contrastive loss
                # Separate embeddings by class
                positive_class_indices = (real_y[:, 1] == 1).nonzero(as_tuple=True)[0]
                negative_class_indices = (real_y[:, 0] == 1).nonzero(as_tuple=True)[0]

                positive_embeddings = mu[positive_class_indices]
                negative_embeddings = mu[negative_class_indices]
                # Create positive pairs
                positive_pairs = torch.norm(positive_embeddings.unsqueeze(1) - positive_embeddings, dim=2)
                # Create negative pairs
                negative_pairs = torch.norm(positive_embeddings.unsqueeze(1) - negative_embeddings, dim=2)

                pos_loss = torch.mean((positive_pairs) ** 2)
                neg_loss = torch.mean(F.relu(1.0 - negative_pairs) ** 2)
                cont_loss = pos_loss + neg_loss

                # Total loss
                loss_val = loss_val + cont_loss*0.001
                loss_val.backward()
                epoch_loss += loss_val.item()  # average per batch
                self.opt.step()

            pbar.set_postfix({"loss": epoch_loss})
            pbar.update(1)

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.best_epoch = epoch
                self.best_encoder = self.encoder
                self.best_decoder = self.decoder
            elif epoch - self.best_epoch > early_stop_thresh:
                print(f"Early stopped training at epoch {epoch} with best model from epoch {self.best_epoch}")
                break

                        # Save model every `save_interval` epochs
            if save_interval !=None and epoch % save_interval == 0:
                torch.save(self.encoder.state_dict(), f"ctencoder_{epoch}.pth")
                torch.save(self.decoder.state_dict(), f"ctdecoder_{epoch}.pth")

        pbar.close()

  
    def check_and_create_loader_con(self, data, batch_size=32):
        if isinstance(data, DataLoader):
            return data
        elif isinstance(data, pd.DataFrame):
            data = data.values.astype(np.float32)
            labels = y_train.values
        elif isinstance(data, np.ndarray):
            if not data.dtype == np.float32:
                data = data.astype(np.float32)
            labels = y_train.values
        else:
            raise ValueError("Data must be a DataLoader, DataFrame, or NumPy array")

        dataset = TensorDataset(torch.from_numpy(data), torch.from_numpy(labels).long())
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_latent_space_con(self, data_loader, aux=True):
        data_loader = self.check_and_create_loader_con(data_loader)
        self.encoder.eval()
        latent_spaces = []
        labels = []
        with torch.no_grad():
            for batch in data_loader:
                real_x, real_y = batch
                real_x = real_x.to(self.device).float()
                real_y = F.one_hot(real_y, num_classes=self.n_classes).to(self.device)
                if aux:
                    mu, _, _ = self.encoder(real_x, real_y)
                else:
                    mu, _ = self.encoder(real_x, real_y)
                latent_spaces.append(mu.cpu().numpy())
                labels.append(real_y.argmax(dim=1).cpu().numpy())
        return np.concatenate(latent_spaces, axis=0), np.concatenate(labels, axis=0)

    @staticmethod
    def generate_distinct_colors(n_colors=10):
        return sns.color_palette("tab10", n_colors=n_colors)

    def plot_latent_space(self, latent_space, labels):
        sns.set_theme(style="whitegrid")
        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)
        colors = self.generate_distinct_colors(n_colors=num_labels)
        color_map = {label: color for label, color in zip(unique_labels, colors)}
        markers = ['o', '^', '*', 's', 'D', 'v', '<', '>']
        marker_map = {label: markers[i % len(markers)] for i, label in enumerate(unique_labels)}
        plt.figure(figsize=(10, 8))
        for label in unique_labels:
            idx = labels == label
            plt.scatter(latent_space[idx, 0], latent_space[idx, 1], c=[color_map[label]], marker=marker_map[label], label=label)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Latent Space Visualization')
        handles = [plt.Line2D([0], [0], marker=marker_map[label], color='w', markerfacecolor=color_map[label], markersize=10) for label in unique_labels]
        plt.legend(handles, unique_labels, title="Classes", loc="best")
        plt.show()

    def save_model(self, path_encoder, path_decoder):
        torch.save(self.best_encoder.state_dict(), path_encoder)
        torch.save(self.best_decoder.state_dict(), path_decoder)

    def load_model(self, path_encoder, path_decoder):
        self.encoder.load_state_dict(torch.load(path_encoder))
        self.decoder.load_state_dict(torch.load(path_decoder))

    def generate_fake_data(self, X_train_enc, y_train, transformer, all=True, unsampler=None):

          self.decoder.eval()

          if all:
            labels = y_train
          else:
            labels = balance_classes(y_train)

          # Ensure labels is a numpy array if it's a pandas DataFrame or Series
          if isinstance(labels, (pd.DataFrame, pd.Series)):
              labels = labels.to_numpy()

          y_train_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
          one_hot_labels = F.one_hot(y_train_tensor, num_classes=self.n_classes).to(self.device)

          mean = torch.zeros(len(labels), embedding_dim)
          std = mean + 1
          noise = torch.normal(mean=mean, std=std).to(self.device)

          fake, sigmas = self.decoder(noise, one_hot_labels)
          fake = torch.tanh(fake)
          raw_data = fake.detach().cpu().numpy()

          X_bal=np.concatenate([X_train_enc,raw_data])
          y_bal=np.concatenate([y_train,labels])

          # if unsampler is not None:
          #     X_bal,y_bal = unsampler.fit_resample(X_bal, y_bal)

          # Perform K-means clustering
          kmeans = KMeans(n_clusters=2)
          clusters = kmeans.fit_predict(X_bal)

          # Separate labels for real and fake data
          real_labels = clusters[:len(X_train_enc)]
          fake_labels = clusters[len(X_train_enc):]

          # Filter fake samples based on cluster dominance
          dominant_cluster = np.argmax(np.bincount(real_labels))
          filtered_X = raw_data[fake_labels == dominant_cluster]
          filtered_y = labels[fake_labels == dominant_cluster]

          X_bal=np.concatenate([X_train_enc,filtered_X])
          y_bal=np.concatenate([y_train,filtered_y])

          if unsampler is not None:
              X_bal,y_bal = unsampler.fit_resample(X_bal, y_bal)

          X_bal = transformer.inverse_transform(X_bal, sigmas.detach().cpu().numpy())

          return X_bal, y_bal