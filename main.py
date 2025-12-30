import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    # 可以在这里做一些全局初始化的工作，比如加载配置
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()