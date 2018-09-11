import math
import cv2
import numpy as np
from scipy import ndimage
from skimage import color
from skimage.measure import regionprops
from skimage.measure import label
from sklearn.datasets import fetch_mldata


# region Vektori
def vektor(b, e):
    x, y = b
    X, Y = e
    return (X - x, Y - y)


def zbirVektora(v, w):
    x, y = v
    X, Y = w
    return (x + X, y + Y)


def skalarniProizvod(v, w):
    x, y = v
    X, Y = w
    return x * X + y * Y


def skaliraj(v, sc):
    x, y = v
    return (sc * x, sc * y)


def intenzitet(v):
    x, y = v
    return math.sqrt(x * x + y * y)


def jedinicni(v):
    x, y = v
    intenz = intenzitet(v)
    return (x / intenz, y / intenz)
# endregion


def ucitajMnist(mnist):
    for i in range(70000):
        slika = mnist.data[i].reshape(28, 28)
        slickica = ((color.rgb2gray(slika) / 255.0) > 0.80).astype('uint8')
        slickica = pozicionirajSliku(slickica)
        mnist_cifre.append(slickica)


def hofova(video):
    capture = cv2.VideoCapture(video)
    kernel = np.ones((2, 2), np.uint8)

    while (capture.isOpened()):
        ret, frejm = capture.read()
        grejskejl = cv2.cvtColor(frejm, cv2.COLOR_BGR2GRAY)

        grejskejl = cv2.dilate(grejskejl, kernel)
        capture.release()

        x0 = 1000
        y0 = 1000
        x1 = -1000
        y1 = -1000
        threshold1 = 50
        threshold2 = 150
        aperture_size = 3
        threshold = 40
        minDuzinaLinije = 100
        maxGapLinije = 8
        ivice = cv2.Canny(grejskejl, threshold1, threshold2, aperture_size)

        linje = cv2.HoughLinesP(ivice, 1, np.pi / 180, threshold, minDuzinaLinije, maxGapLinije)

        for lin in linje:
            x01 = lin[0][0]
            y01 = lin[0][1]
            x02 = lin[0][2]
            y02 = lin[0][3]

            if x01 < x0:
                y0 = y01
                x0 = x01
            if x02 > x1:
                x1 = x02
                y1 = y02

        return x0, y0, x1, y1


def projTackuNaDuz(tacka, pocetak, kraj):
    linijaV = vektor(pocetak, kraj)
    tackaV = vektor(pocetak, tacka)
    duzinaLinije = intenzitet(linijaV)
    duzJedinicni = jedinicni(linijaV)
    skaliranaTackaV = skaliraj(tackaV, 1.0 / duzinaLinije)
    t = skalarniProizvod(duzJedinicni, skaliranaTackaV)
    r = 1
    if t < 0.0:
        t = 0.0
        r = -1
    elif t > 1.0:
        t = 1.0
        r = -1
    nearest = skaliraj(linijaV, t)
    dist = intenzitet(vektor(nearest, tackaV))
    nearest = zbirVektora(nearest, pocetak)
    return (dist, (int(nearest[0]), int(nearest[1])), r)


def pozicionirajSliku(slikaCB):
    slika = np.zeros((28, 28), np.uint8)
    x1 = -1000
    x2 = 1000
    y1 = -1000
    y2 = 1000

    try:
        labeliranaSlika = label(slikaCB)
        regije = regionprops(labeliranaSlika)
        for regija in regije:
            okvir = regija.bbox
            if okvir[0] < x2:
                x2 = okvir[0]
            if okvir[1] < y2:
                y2 = okvir[1]
            if okvir[2] > x1:
                x1 = okvir[2]
            if okvir[3] > y1:
                y1 = okvir[3]

        poHorizontali = x1 - x2
        poVertikali = y1 - y2
        slika[0: poHorizontali, 0: poVertikali] = slikaCB[x2: x1, y2: y1]
        return slika
    except ValueError:
        pass


def prepoznajCifru(slika):
    slikaCB = ((color.rgb2gray(slika) / 255.0) > 0.90).astype('uint8')
    slika = pozicionirajSliku(slikaCB)
    minRazlika = 10000
    rez = -1
    for i in range(len(mnist_cifre)):
        mnistSlika = mnist_cifre[i]
        brRazlika = np.sum(mnistSlika != slika)
        if brRazlika < minRazlika:
            minRazlika = brRazlika
            rez = mnist.target[i]
    return rez


def prodjiKrozFrejmove():
    frejm = 0
    while (1):
        ret, slika = video0.read()
        if not ret:
            break

        maska = cv2.inRange(slika, donja, gornja)

        slikaCB = maska * 1.0
        slikaCB2 = slikaCB

        slikaCB = cv2.dilate(slikaCB, kernel)

        slikaCBLabel, niz = ndimage.label(slikaCB)
        objekti = ndimage.find_objects(slikaCBLabel)

        for i in range(niz):
            centar = []
            duzina = []
            lokacija = objekti[i]

            duzina.append(lokacija[1].stop - lokacija[1].start)
            duzina.append(lokacija[0].stop - lokacija[0].start)

            centar.append((lokacija[1].stop + lokacija[1].start) / 2)
            centar.append((lokacija[0].stop + lokacija[0].start) / 2)

            if duzina[0] > 10 or duzina[1] > 10:
                cifra = {'centar': centar, 'duzina': duzina, 'frejm': frejm}

                rez = [] # cifre koje su blizu
                for cif in cifre:
                    if (tolerancija > intenzitet(vektor(cifra['centar'], cif['centar']))):
                        rez.append(cif)

                if len(rez) == 0:
                    x11 = centar[0] - 14
                    y11 = centar[1] - 14
                    x22 = centar[0] + 14
                    y22 = centar[1] + 14
                    global id
                    id += 1
                    cifra['id'] = id
                    cifra['prosao'] = False
                    cifra['vrednost'] = prepoznajCifru(slikaCB2[int(y11):int(y22), int(x11):int(x22)])
                    cifra['slika'] = slikaCB2[int(y11):int(y22), int(x11):int(x22)]
                    cifre.append(cifra)
                else:
                    najbliziElement = rez[0]
                    min = intenzitet(vektor(najbliziElement['centar'], cifra['centar']))
                    for el in rez:
                        udaljenost = intenzitet(vektor(el['centar'], cifra['centar']))
                        if udaljenost < min:
                            najbliziElement = el
                            min = udaljenost
                    cif = najbliziElement
                    cif['centar'] = cifra['centar']
                    cif['frejm'] = cifra['frejm']

        for cif in cifre:
            if (frejm - cif['frejm'] < 3):
                dist, pnt, r = projTackuNaDuz(cif['centar'], ivice[0], ivice[1])
                if r > 0:
                    if dist < 10:
                        if not cif['prosao']:
                            cif['prosao'] = True
                            global zbir
                            zbir += cif['vrednost']
                            print (format(int(cif['vrednost'])))

        frejm += 1


outString = "RA 33/2013 Obren Starovic\nfile\tsum"
tolerancija = 15
for i in range(10):
    zbir = 0
    video = "video-" + str(i) + ".avi"
    print(video)
    video0 = cv2.VideoCapture(video)

    x1, y1, x2, y2 = hofova(video)

    ivice = [(x1, y1), (x2, y2)]

    mnist = fetch_mldata('MNIST original')
    mnist_cifre = []
    id = -1
    ucitajMnist(mnist)

    cifre = []

    donja = np.array([160, 160, 160], dtype="uint8")
    gornja = np.array([255, 255, 255], dtype="uint8")
    kernel = np.ones((2, 2), np.uint8)
    prodjiKrozFrejmove()
    print ("zbir: " + format(int(zbir)))
    video0.release()

    outString += "\n" + video + "\t" + str(int(zbir))

f = open('out.txt', 'w')
f.write(outString)
f.close()

print(outString)
