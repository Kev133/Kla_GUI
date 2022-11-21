import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import numpy as np
from PyQt5 import QtCore
import PyQt5.QtGui as qtg
from PyQt5.QtWidgets import QWidget, QLabel, QFileDialog, QPushButton, QDialog,QFontDialog,QStyle,\
    QPlainTextEdit,QMessageBox,QApplication,QRadioButton,QHBoxLayout,QGroupBox,QVBoxLayout
import sys
import vypocet

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import time
import pandas as pd
import openpyxl

class Hlavni(qtw.QMainWindow):
    def __init__(self):
        qtw.QMainWindow.__init__(self)
        # definovani vlastnosti pro program
        self.name = None
        self.namerene = None
        self.char_sondy = None
        self.model_valuesN = []
        self.namereneN = []
        self.init_ui()

    def init_ui(self):
        # graficke veci
        #TODO zmenit celkovy vzhled programu aby vypadal moderneji a intiutivně (hlavne asi font a nejake rozlozeni)

        self.setWindowTitle("  Vyhodnocení dat z kyslíkové sondy")
        self.setFont(qtg.QFont("Arial",11))
        self.setGeometry(200, 200, 900, 500)
        self.setWindowIcon(qtg.QIcon('kla_icon.png'))
        self.setStyleSheet("")

        self.button1 = QPushButton(" Přidat adresář", self)
        self.button1.setGeometry(430, 20, 130, 40)
        self.button1.clicked.connect(self.pruzkum)
        # ikona u přidat adresar
        ikonka = QStyle.SP_DirOpenIcon
        icon = self.style().standardIcon(ikonka)

        self.button1.setIcon(icon)

        # self.button_help = QPushButton("Help", self)
        # self.button_help.clicked.connect(self.help_okno)
        # self.button_help.setGeometry(850, 0, 50, 30)

        self.button2 = QPushButton("Proveď výpočet", self)
        self.button2.setGeometry(430, 65, 130, 40)
        self.button2.clicked.connect(self.nacteni_dat)

        #radiobutton
        self.group_box = QGroupBox(self)
        self.group_box.setTitle("Optimalizační metoda")
        self.group_box.setFont(qtg.QFont("Arial",11))
        # vbox = QVBoxLayout()
        # vbox.addWidget(self.group_box)
        self.group_box.setGeometry(70,10,330,100)
        self.hbox_layout = QHBoxLayout ()

        self.opt_min1 = QRadioButton("Nelder-Mead  ",self)
        self.opt_min2 = QRadioButton("BFGS",self)
        self.opt_min3 = QRadioButton("Powell",self)
        self.opt_min1.setChecked(True)

        self.hbox_layout.addWidget(self.opt_min1)
        self.hbox_layout.addWidget(self.opt_min2)
        self.hbox_layout.addWidget(self.opt_min3)

        self.opt_min1.clicked.connect(self.nelder)
        self.opt_min2.clicked.connect(self.bfgs)
        self.opt_min3.clicked.connect(self.powell)

        self.group_box.setLayout(self.hbox_layout)
        self.group_box.setFont(qtg.QFont("Arial", 11))

        self.info_for_user = QPlainTextEdit("Vítejte v programu na výpočet kLa.\n\n\n", self)
        self.info_for_user.setGeometry(580, 20, 270, 440)
        self.info_for_user.setReadOnly(True)


        #zavede cistou canvas pro graf
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        self.canvas.setGeometry(20, 120, 550, 340)
        self.fig_ax = self.figure.add_subplot(111)
        self.figure.set_tight_layout(True)

        self.radio_button_value = 1

        self.show()

    #funkce pro vyber optimalizacni metody, cislo je pak passed do vypocet.Optimalizace.opt()
    def nelder(self):
        self.radio_button_value = 1
    def bfgs(self):
        self.radio_button_value = 2
    def powell(self):
        self.radio_button_value = 3


    def pruzkum(self):
        name = QFileDialog.getExistingDirectory(self)
        if name:
            self.name = name
            self.info_for_user.appendPlainText("Data pro výpočet jsou v souboru\n"+name+"\n")


    def help_okno(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Help")
        dlg.setGeometry(300,300,600,200)
        dlg.exec_()
        #TODO napsat do help okna nejakej text vysvetlujici co program dela
        # a co ma delat uzivatel (nahraj soubory konstant,paramy atd do jedne slozky,
        # tu najdi v adresari a spust vypocet) neco takoveho (jako "readme" u matlab prog)
        # Hlavne to nejak napsat pres QPlainTextEdit


    def nacteni_dat(self):
        """
        Tato funkce nahraje data do objektů namerene a char_sondy

        """
        if self.name == None:
            QMessageBox.warning(self,"Warning","Nebyla přidána složka s daty")
        else:

            #nacteni dat v souboru namerene_hodnoty
            with open(self.name+"/namerene_hodnoty.txt", "r") as f:
                hodnoty1 = f.read().splitlines()
            self.namerene = list(map(float, hodnoty1))

            with open(self.name+"/konstant.txt", "r") as f:
                hodnoty2 = f.read().splitlines()
            self.char_sondy = list(map(float, hodnoty2))

            # info do GUI
            self.info_for_user.appendPlainText("Data načtena z konstant.txt a namerene_hodnoty.txt\n")
            QtCore.QCoreApplication.processEvents()
            try:    # kdyz jsou data nactena, spusti se funkce vypocet
                self.vypocet()

            except Exception as e:
                print(str(e))
                self.info_for_user.appendPlainText("V této složce nejsou vhodné soubory pro výpočet.\nZvolte správnou složku.")

    def vypocet(self):
        self.start_time=time.time()
        #self.info_for_user.appendPlainText(f"Čas začátku výpočtu: {self.start_time} s")
        self.vysledek = vypocet.Optimalizace(self.char_sondy,self.namerene).opt(self.radio_button_value)
        self.end_time=time.time()
        self.elapsed_time=self.end_time-self.start_time
        print(self.elapsed_time)
        #self.info_for_user.appendPlainText(f"Čas konce výpočtu: {self.end_time}\n")
        self.elapsed_time = self.end_time - self.start_time
        self.info_for_user.appendPlainText(f"Čas výpočtu: {round(self.elapsed_time,2)} s\n")
        QtCore.QCoreApplication.processEvents()
        #vypocet.Optimalizace(self.char_sondy,self.namerene).graph()
        self.kla = float(self.vysledek)
        self.info_for_user.appendPlainText(f"Hodnota kLa je: {round(self.kla,7)} s\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}\n")
        self.plotni_to()
        QMessageBox.information(self, "Hotovo",f"kLa je: {round(self.kla,5)} s\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}\n"
                                       f"Čas výpočtu: {round(self.elapsed_time,2)} s")
    def plotni_to(self):
        tau = np.linspace(0, 100, num=1000)
        self.model_values = np.exp(-self.kla * tau)
        for i in range(0, len(self.model_values)):
            self.model_valuesN.append(
                (self.model_values[i] - max(self.model_values)) / (min(self.model_values) - max(self.model_values)))

        for i in range(0, len(self.namerene)):
            self.namereneN.append(
                (self.namerene[i] - min(self.namerene)) / (max(self.namerene) - min(self.namerene)))


        self.fig_ax.plot(self.namereneN[0:int(len(self.namereneN)/2)])
        self.fig_ax.plot(self.char_sondy)
        self.fig_ax.plot(self.model_valuesN)
        self.fig_ax.set_xlabel("čas [s]")
        self.fig_ax.set_ylabel("dx0\N{SUBSCRIPT TWO} normalizované")
        self.fig_ax.margins(x=0)
        self.fig_ax.legend(["naměřené hodnoty","impulzní charakteristika","funkce z optimalizace"])

        self.canvas.draw()
        self.ulozeni_dat()
        # add toolbar
        self.addToolBar(Qt.BottomToolBarArea, NavigationToolbar(self.canvas, self))
    def ulozeni_dat(self):
        vyp_excel=pd.DataFrame([self.kla],index=["měření č.1"],columns=["kLa"])
        with pd.ExcelWriter(self.name+ "\kla_vysledky.xlsx") as writer:
           vyp_excel.to_excel(writer,sheet_name="vysledky")

        self.info_for_user.appendPlainText("Výsledek je uložen v excel souboru kla_vysledky.xlsx")


def main():

    app = qtw.QApplication([])
    gui = Hlavni()
    app.exec_()


if __name__=="__main__":
    main()
