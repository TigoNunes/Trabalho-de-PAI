import tkinter 
janela = tkinter.Tk()
janela.title("Trabalho de PAI")
janela.minsize(width= 500, height=300)
janela.config(padx=25, pady= 25, bg= "White")

my_label = tkinter.Label(text="Rotulo", font = ("Arial", 24, "bold"))  
my_label.pack() #mais estático 

def button_clicked():
    my_label["text"] = input.get()

button = tkinter.Button(text= "Botão 1", command= button_clicked)
button.pack()


janela.mainloop()