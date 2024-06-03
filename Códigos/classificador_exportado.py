# %% [markdown]
# Importações

# %%
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import moments_hu
from skimage.color import rgb2hsv

# %% [markdown]
# Diretorios e tratamentos das planilias 

# %%
# Caminho para o diretório de imagens e para o arquivo CSV
image_dir = 'F:/6º Periodo/PAI/Trabalho-de-PAI/dataset'
csv_path = 'F:/6º Periodo/PAI/Trabalho-de-PAI/classifications.csv'

# %%
df = pd.read_csv(csv_path)
df

# %%
df_filtrado = df[df['image_filename'].isin(os.listdir('F:/6º Periodo/PAI/Trabalho-de-PAI/dataset'))]
# df_filtrado.to_csv("filtered_file.csv", index=False)
df_filtrado

# %% [markdown]
# Recortando as Imagens e Salvando nas pastas

# %%
def crop_subimage_with_padding(image, x, y, size=100):
    half_size = size // 2

    # Coordenadas para o corte
    left = x - half_size
    upper = y - half_size
    right = x + half_size
    lower = y + half_size

    # Verificar se as coordenadas estão fora dos limites da imagem e ajustar com padding
    padding_left = max(0, -left)
    padding_top = max(0, -upper)
    padding_right = max(0, right - image.width)
    padding_bottom = max(0, lower - image.height)

    # Ajustar coordenadas de corte para estarem dentro dos limites da imagem
    left = max(0, left)
    upper = max(0, upper)
    right = min(image.width, right)
    lower = min(image.height, lower)

    # Recortar a imagem
    cropped_image = image.crop((left, upper, right, lower))

    # Adicionar padding se necessário
    if padding_left > 0 or padding_top > 0 or padding_right > 0 or padding_bottom > 0:
        cropped_image = ImageOps.expand(cropped_image, border=(padding_left, padding_top, padding_right, padding_bottom), fill=0)

    # Garantir que a subimagem tenha exatamente 100x100 pixels
    cropped_image = cropped_image.resize((size, size))
    
    return cropped_image

# %%
# Diretório para salvar as subimagens
subimage_dir = 'F:/6º Periodo/PAI/Trabalho-de-PAI/sub-imagens/'
os.makedirs(subimage_dir, exist_ok=True)

# %%
atributos_classe_cell = df_filtrado['bethesda_system'].unique()
print(atributos_classe_cell)
for atributo in atributos_classe_cell:
    x = f'F:/6º Periodo/PAI/Trabalho-de-PAI/sub-imagens/{atributo}'
    os.makedirs(x, exist_ok=True)

# %%
for index, row in df_filtrado.iterrows():
    try:
        filename = row['image_filename']
        x, y = int(row['nucleus_x']), int(row['nucleus_y'])
        classe_celula = row['bethesda_system']
        
        # Caminho completo para a imagem
        image_path = os.path.join(image_dir, filename)
        
        # Carregar a imagem
        image = Image.open(image_path).convert('RGB')
        # Recortar a subimagem
        subimage = crop_subimage_with_padding(image, x, y)
        
        # Salvar a subimagem
        subimage_filename = f"{row['cell_id']}.png"
        subimage_path = os.path.join(subimage_dir,classe_celula,subimage_filename)
        subimage.save(subimage_path)

        # print(f'Subimagem salva em {subimage_path}')
    except:
        pass
    

# %% [markdown]
# Histograms

# %%
def reduce_grayscale_depth(image, levels=16):
    """ Reduz a profundidade de tons de cinza para o número especificado de níveis. """
    factor = 256 // levels
    reduced_image = image // factor
    return reduced_image

# %%
def plot_histogram(image, levels=16):
    """ Gera e plota o histograma da imagem com a profundidade reduzida de tons de cinza. """
    # Reduzir a profundidade dos tons de cinza
    reduced_image = reduce_grayscale_depth(np.array(image), levels)
    
    # Calcular o histograma
    histogram, bin_edges = np.histogram(reduced_image, bins=levels, range=(0, levels))

    # Plotar o histograma
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], histogram, width=0.8, color='gray')
    plt.xlabel('Nível de Cinza')
    plt.ylabel('Frequência')
    plt.title('Histograma de Tons de Cinza com 16 Níveis')
    plt.xticks(bin_edges[:-1])
    plt.show()

# %%
# Caminho da subimagem para análise
subimage_path = 'F:/6º Periodo/PAI/Trabalho-de-PAI/sub-imagens/ASC-H/8140.png'

# Carregar a subimagem
subimage = Image.open(subimage_path).convert('L')

# Gerar e plotar o histograma
plot_histogram(subimage, levels=16)

# %% [markdown]
# Implemente a funcionalidade de geração dos histogramas de cor da imagem com quantização de
# 16 valores para o canal H e 8 valores para o V (histograma 2D com 16*8 entradas).

# %%
def quantize_channel(channel, levels):
    """ Quantiza o canal de uma imagem para o número especificado de níveis. """
    factor = 256 // levels
    quantized_channel = channel // factor
    return quantized_channel

# %%
def plot_histogram_2d(image_path, h_levels=16, v_levels=8):
    """ Gera e plota o histograma 2D para os canais H e V com quantização especificada. """
    # Carregar a imagem
    subimage = Image.open(image_path).convert('RGB')
    
    # Converter a imagem para o espaço de cores HSV
    hsv_image = subimage.convert('HSV')
    hsv_array = np.array(hsv_image)
    
    # Separar os canais H, S e V
    h_channel, s_channel, v_channel = hsv_array[:,:,0], hsv_array[:,:,1], hsv_array[:,:,2]
    
    # Quantizar os canais H e V
    h_quantized = quantize_channel(h_channel, h_levels)
    v_quantized = quantize_channel(v_channel, v_levels)
    
    # Calcular o histograma 2D
    histogram, x_edges, y_edges = np.histogram2d(h_quantized.flatten(), v_quantized.flatten(), bins=[h_levels, v_levels], range=[[0, h_levels], [0, v_levels]])

    # Plotar o histograma 2D
    plt.figure(figsize=(10, 6))
    plt.imshow(histogram.T, origin='lower', cmap='gray', aspect='auto')
    plt.colorbar()
    plt.xlabel('Quantização do Canal H (Hue)')
    plt.ylabel('Quantização do Canal V (Value)')
    plt.title('Histograma 2D Quantizado para Canais H e V')
    plt.xticks(np.arange(h_levels), labels=np.arange(h_levels))
    plt.yticks(np.arange(v_levels), labels=np.arange(v_levels))
    plt.show()

# %%
# Caminho da subimagem para análise
subimage_path = 'F:/6º Periodo/PAI/Trabalho-de-PAI/sub-imagens/ASC-H/8140.png'

# Gerar e plotar o histograma 2D
plot_histogram_2d(subimage_path, h_levels=16, v_levels=8)

# %% [markdown]
# Calcular as matrizes de co-ocorrência Ci,i onde i=1,2,4,8,16 e 32, considerando 16 tons de cinza

# %%
def calculate_cooccurrence_matrices(image, levels=16, distances=[1, 2, 4, 8, 16, 32]):
    """ Calcula as matrizes de co-ocorrência para diferentes distâncias. """
    # Reduzir a profundidade dos tons de cinza
    reduced_image = reduce_grayscale_depth(np.array(image), levels)
    
    # Calcular matrizes de co-ocorrência
    cooccurrence_matrices = {}
    for distance in distances:
        matrix = graycomatrix(reduced_image, [distance], [0], levels=levels, symmetric=True, normed=True)
        cooccurrence_matrices[distance] = matrix
    return cooccurrence_matrices

# %%
def plot_cooccurrence_matrices(cooccurrence_matrices):
    """ Exibe as matrizes de co-ocorrência calculadas. """
    num_matrices = len(cooccurrence_matrices)
    fig, axes = plt.subplots(1, num_matrices, figsize=(20, 5))
    
    for ax, (distance, matrix) in zip(axes, cooccurrence_matrices.items()):
        ax.imshow(matrix[:, :, 0, 0], cmap='gray', interpolation='nearest')
        ax.set_title(f'Distância = {distance}')
        ax.set_xlabel('Níveis de Cinza')
        ax.set_ylabel('Níveis de Cinza')
    
    plt.tight_layout()
    plt.show()

# %%
# Caminho da subimagem para análise
subimage_path = 'F:/6º Periodo/PAI/Trabalho-de-PAI/sub-imagens/HSIL/7604.png'

# Carregar a subimagem
subimage = Image.open(subimage_path).convert('L')

# Calcular as matrizes de co-ocorrência
distances = [1, 2, 4, 8, 16, 32]
cooccurrence_matrices = calculate_cooccurrence_matrices(subimage, levels=16, distances=distances)

# Exibir as matrizes de co-ocorrência
plot_cooccurrence_matrices(cooccurrence_matrices)

# %% [markdown]
# Calcular os descritores de Haralick Entropia, Homogeneidade e Contraste para as matrizes de co-
# ocorrência do item anterior (3*6 características

# %%
def calculate_entropy(matrix):
    """ Calcula a entropia a partir da matriz de co-ocorrência normalizada. """
    # Evitar log de zero
    non_zero = matrix[matrix > 0]
    entropy = -np.sum(non_zero * np.log2(non_zero))
    return entropy

# %%
def calculate_haralick_descriptors(cooccurrence_matrices):
    """ Calcula os descritores de Haralick para as matrizes de co-ocorrência fornecidas. """
    properties = ['contrast', 'homogeneity']
    descriptors = {prop: [] for prop in properties}
    descriptors['entropy'] = []
    
    for distance, matrix in cooccurrence_matrices.items():
        for prop in properties:
            value = graycoprops(matrix, prop)[0, 0]
            descriptors[prop].append(value)
        entropy = calculate_entropy(matrix[:, :, 0, 0])
        descriptors['entropy'].append(entropy)
    
    return descriptors

# %%
# Calcular os descritores de Haralick
haralick_descriptors = calculate_haralick_descriptors(cooccurrence_matrices)

# Exibir os descritores
for prop, values in haralick_descriptors.items():
    print(f"{prop.capitalize()}:")
    for distance, value in zip(distances, values):
        print(f"  Distância {distance}: {value:.4f}")

# %% [markdown]
# Calcular os momentos invariantes de Hu para a imagem em 256 tons de cinza e para os 3 canais
# originais do modelo HSV (4*7 características)

# %%
def calculate_hu_moments(image):
    """ Calcula os momentos invariantes de Hu para uma imagem. """
    moments = moments_hu(image)
    return moments

# %%
subimage_256 = Image.open(subimage_path).convert('L')
subimage_256_np = np.array(subimage_256)
hu_moments_grayscale = calculate_hu_moments(subimage_256_np)

# Calcular momentos de Hu para os canais HSV
subimage_rgb = Image.open(subimage_path).convert('RGB')
subimage_hsv = rgb2hsv(np.array(subimage_rgb))
hu_moments_h = calculate_hu_moments(subimage_hsv[:, :, 0])
hu_moments_s = calculate_hu_moments(subimage_hsv[:, :, 1])
hu_moments_v = calculate_hu_moments(subimage_hsv[:, :, 2])

# Exibir os momentos de Hu
print("Momentos de Hu para a imagem em 256 tons de cinza:")
for i, moment in enumerate(hu_moments_grayscale, start=1):
    print(f"  Hu[{i}]: {moment:.4e}")

print("Momentos de Hu para o canal H do modelo HSV:")
for i, moment in enumerate(hu_moments_h, start=1):
    print(f"  Hu[{i}]: {moment:.4e}")

print("Momentos de Hu para o canal S do modelo HSV:")
for i, moment in enumerate(hu_moments_s, start=1):
    print(f"  Hu[{i}]: {moment:.4e}")

print("Momentos de Hu para o canal V do modelo HSV:")
for i, moment in enumerate(hu_moments_v, start=1):
    print(f"  Hu[{i}]: {moment:.4e}")


