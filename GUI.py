import PyQt5.QtWidgets as qtw
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import PyQt5.QtGui as qtg
from PyQt5.QtWidgets import QWidget, QLabel, QFileDialog, QPushButton, QDialog,QFontDialog,QStyle,QPlainTextEdit,QMessageBox
import sys
import vypocet



class Hlavni(qtw.QMainWindow):
    def __init__(self):
        qtw.QMainWindow.__init__(self)
        # definovani vlastnosti pro program
        self.name = None
        self.namerene = None
        self.char_sondy = None

        self.init_ui()

    def init_ui(self):
        # graficke veci
        #TODO zmenit celkovy vzhled programu aby vypadal moderneji a intiutivně (hlavne asi font a nejake rozlozeni)

        self.setWindowTitle("  Vyhodnocení dat z kyslíkové sondy")

        self.setGeometry(200, 200, 900, 580)
        self.setWindowIcon(qtg.QIcon('kla_icon.png'))
        self.setStyleSheet("")
        self.button1 = QPushButton(" Přidat adresář", self)
        self.button1.setGeometry(250, 100, 110, 30)
        self.button1.clicked.connect(self.pruzkum)
        # ikona u přidat adresar
        ikonka = QStyle.SP_DirOpenIcon
        icon = self.style().standardIcon(ikonka)

        self.button1.setIcon(icon)

        self.file_label = QLabel("               Kla je:  ", self)
        self.file_label.setGeometry(10, 100, 230, 30)

        self.button_help = QPushButton("Help", self)
        self.button_help.clicked.connect(self.help_okno)
        self.button_help.setGeometry(850, 0, 50, 30)

        self.button2 = QPushButton("Proveď výpočet", self)
        self.button2.setGeometry(250, 140, 100, 30)
        self.button2.clicked.connect(self.nacteni_dat)

        self.info_for_user = QPlainTextEdit("Vítejte v programu na výpočet kLa.\nV tomto okně se budou vypisovat informace o tom, co program dělá\n\n", self)
        self.info_for_user.setGeometry(520, 50, 300, 400)
        self.info_for_user.setReadOnly(True)

        self.show()

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
        Tato funkce by mela byt docela foolproof, nahraje data do objektů namerene a char_sondy

        """
        if self.name == None:
            QMessageBox.warning(self,"Warning","Nebyla přidána složka s daty")
        else:
            try:
                #nacteni dat v souboru namerene_hodnoty
                with open(self.name+"/namerene_hodnoty.txt", "r") as f:
                    hodnoty1 = f.read().splitlines()
                self.namerene = list(map(float, hodnoty1))

                with open(self.name+"/konstant.txt", "r") as f:
                    hodnoty2 = f.read().splitlines()
                self.char_sondy = list(map(float, hodnoty2))

                # info do GUI
                self.info_for_user.appendPlainText("Data načtena z konstant.txt a namerene_hodnoty.txt\n")
                # kdyz jsou data nactena, spusti se funkce vypocet
                self.vypocet()

            except Exception as e:
                print(str(e))
                self.info_for_user.appendPlainText("V této složce nejsou vhodné soubory pro výpočet.\nZvolte správnou složku.")

    def vypocet(self):
        self.vysledek = vypocet.Optimalizace(self.char_sondy,self.namerene).vysledek.x
        self.info_for_user.appendPlainText("Úspěšně vypočteno\n")
        vypocet.Optimalizace(self.char_sondy,self.namerene).graph()
        self.file_label.setText(str(self.vysledek))
        self.ulozeni_dat()

    def ulozeni_dat(self):

        with open(self.name+"/vysledky.txt", 'w') as f:
                for line in self.vysledek:
                    f.write(f"{line}\n")

        self.info_for_user.appendPlainText("Výsledek je uložen v souboru vysledeky.txt.\n\nDěkujeme za využití našeho programu\n\n\nS pozdravem\nHlavní vývojář Kevin Klee")





def main():

    app = qtw.QApplication([])
    gui = Hlavni()
    app.exec_()


if __name__=="__main__":
    main()
