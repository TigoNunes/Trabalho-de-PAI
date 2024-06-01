import os
import tkinter as tk
from PIL import Image, ImageTk

# Variável global para armazenar a imagem original
imagem_original = None
zoom_factor = 1.0
pasta_imagens = "C:/Users/Gabriel/Desktop/Trabalho-de-PAI-main/dataset"

# Função para listar imagens na pasta
def listar_imagens(pasta):
    return [os.path.basename(f) for f in os.listdir(pasta) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# Função para carregar e exibir a imagem
def carregar_imagem(caminho_imagem=None):
    global imagem_original, zoom_factor
    if caminho_imagem:
        imagem_original = Image.open(caminho_imagem)
        zoom_factor = calcular_zoom_inicial(imagem_original)  # Definir fator de zoom inicial
    if imagem_original:
        largura, altura = imagem_original.size
        nova_largura = int(largura * zoom_factor)
        nova_altura = int(altura * zoom_factor)
        imagem_redimensionada = imagem_original.resize((nova_largura, nova_altura))
        imagem_tk = ImageTk.PhotoImage(imagem_redimensionada)
        canvas_imagem.create_image(0, 0, anchor=tk.NW, image=imagem_tk)
        canvas_imagem.image = imagem_tk  # Necessário para evitar que a imagem seja coletada pelo garbage collector
        canvas_imagem.config(scrollregion=canvas_imagem.bbox(tk.ALL))

# Função para calcular o zoom inicial
def calcular_zoom_inicial(imagem):
    largura, altura = imagem.size
    max_dim = 595  # Tamanho máximo desejado para a exibição inicial
    fator_zoom = min(max_dim / largura, max_dim / altura, 1.0)
    return fator_zoom

# Função para aplicar o zoom na imagem
def aplicar_zoom(event=None, fator=None):
    global zoom_factor
    if fator:
        zoom_factor *= fator
    elif event:
        if event.delta > 0:
            zoom_factor *= 1.1  # Aumentar zoom
        else:
            zoom_factor /= 1.1  # Diminuir zoom
    carregar_imagem()

# Função para lidar com a seleção de uma imagem na Listbox
def selecionar_imagem(event):
    selecionado = listbox_imagens.curselection()
    if selecionado:
        caminho_imagem = os.path.join(pasta_imagens, listbox_imagens.get(selecionado))
        carregar_imagem(caminho_imagem)


# Criação da janela principal
janela = tk.Tk()
janela.title("Visualizador de Imagens")
janela.geometry("750x750")
janela.config(padx=5, pady=5)
janela.tk_setPalette(background='#616161', foreground='white', activeBackground='#212121', activeForeground='white')

# Frame para a lista de imagens
frame_lista = tk.Frame(janela)
frame_lista.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
# Listbox para exibir os nomes das imagens
lista_imagens = listar_imagens(pasta_imagens)
listbox_imagens = tk.Listbox(frame_lista)
listbox_imagens.pack(fill=tk.BOTH, expand=True)
for imagem in lista_imagens:
    listbox_imagens.insert(tk.END, imagem)
listbox_imagens.bind('<<ListboxSelect>>', selecionar_imagem)

# Frame para a imagem
frame_imagem = tk.Frame(janela)
frame_imagem.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
# Canvas para exibir a imagem com barras de rolagem
canvas_imagem = tk.Canvas(frame_imagem)
canvas_imagem.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Frame para os botões
frame_botoes = tk.Frame(janela)
frame_botoes.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
# Botões na área
for i in range(8):
    botao = tk.Button(frame_botoes, text=f"Botão {i+1}")
    botao.pack(side=tk.LEFT, padx=5, pady=5)

# Bind do evento de rolagem do mouse para aplicar zoom
canvas_imagem.bind("<MouseWheel>", aplicar_zoom)
# Iniciar o loop principal da interface
janela.mainloop()
