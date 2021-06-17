import os
import sys
import time

from PyQt5.QtWidgets import (QMdiSubWindow,
                             QMainWindow,
                             QApplication,
                             QPushButton,
                             QLineEdit,
                             QMessageBox,
                             QListWidget, QListWidgetItem,
                             QFrame, QVBoxLayout, QFormLayout, QSizePolicy, QSizeGrip,
                             QGridLayout, QDesktopWidget, QComboBox, QPlainTextEdit
                             )

from PyQt5.QtCore import pyqtSlot, QUrl, Qt
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QFont, QColor

from requester.requester import Requester
from requester.noisyRequester import NoisyRequester

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'My search engine'
        self.k = 428
        self.Requester = Requester(k=self.k)
        self.k_accuracies = [int(k) for k in os.listdir("./parsing/parsed/5_120000_wikipedia1/svd")]
        self.k_accuracies.sort(reverse=True)
        self.k_accuracies = [str(k) for k in self.k_accuracies]
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(350, 150, 1200, 800)
        self.setLayout(QFormLayout())
        self.resizeEvent = self.resize_


        # Create textbox for query
        self.textbox = QLineEdit(self)
        self.textbox.setGeometry(20, 40, 450, 40)
        self.textbox.setFont(QFont("Times", 15))
        self.textbox.returnPressed.connect(self.on_click_search)

        # Accuracy changing
        self.accuracy_k_list = QComboBox(self)
        self.accuracy_k_list.addItems(self.k_accuracies)
        self.accuracy_k_list.addItem("Noisy")
        self.accuracy_k_list.setCurrentText(str(self.k))

        self.button = QPushButton('change k', self)
        self.button.setFont(QFont("Times", 12))
        self.button.setGeometry(110, 0, 90, 30)
        self.button.clicked.connect(self.on_click_change_k)

        self.k_text_field = QPlainTextEdit("current k=" + str(self.k), self)
        self.k_text_field.setGeometry(210, 0, 170, 35)
        self.k_text_field.setFont(QFont("Times", 12))
        self.k_text_field.setReadOnly(True)
        self.k_change_click_counter = 0


        # Create a search button
        self.button = QPushButton('search', self)
        self.button.setFont(QFont("Times", 12))
        self.button.setGeometry(20, 90, 70, 30)
        # connect button to function on_click
        self.button.clicked.connect(self.on_click_search)


        # List of found links
        self.req_res_list = QListWidget(self)
        self.req_res_list.setGeometry(20, 160, 450, 600)
        self.req_res_list.itemSelectionChanged.connect(self.on_click_load_website)
        self.req_res_list.setFont(QFont("Times", 15))


        # Text browser
        frm_webui = QFrame(self)
        self.frm_webui = frm_webui
        frm_webui.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        frm_webui.setGeometry(500, 20, 650, 720)
        webui = QWebEngineView()
        self.webui = webui
        self.layout().addWidget(frm_webui)

        frm_webui.setFrameShape(QFrame.StyledPanel)
        frm_webui.setLayout(QVBoxLayout())
        frm_webui.layout().addWidget(webui)

        frm_webui.setWindowState(Qt.WindowMaximized)



        self.show()


    @pyqtSlot()
    def on_click_search(self):
        textboxValue = self.textbox.text()
        Query_res = self.Requester.make_query(textboxValue, 200)
        self.req_res_list.clear()
        for i, (u,t) in enumerate(Query_res):
            item = QListWidgetItem(f"{i}: {t}")
            item.setData(Qt.UserRole, u)
            self.req_res_list.addItem(item)


    @pyqtSlot()
    def on_click_load_website(self):
        url = self.req_res_list.currentItem().data(Qt.UserRole)
        print(url)
        self.webui.load(QUrl(url))

    @pyqtSlot()
    def on_click_change_k(self):
        self.k_change_click_counter = (self.k_change_click_counter + 1) % 4
        try:
            k = self.accuracy_k_list.currentText()
            if k == "Noisy":
                self.k_text_field.setPlainText("Using noisy requester")
                if self.k_change_click_counter % 2 == 1:
                    del self.Requester
                    self.Requester = None
                    self.Requester = NoisyRequester()
                    self.k = 0
            else:
                self.k_text_field.setPlainText("current k=" + str(self.k))
                if self.k_change_click_counter % 2 == 1:
                    del self.Requester
                    self.Requester = Requester(k=int(k))
                    self.k = int(k)
        except Exception as e:
            print(e)




    @pyqtSlot()
    def resize_(self, event):
        self.frm_webui.resize(self.frameGeometry().width() - (self.textbox.width() + 100), self.frameGeometry().height() - 100)

