from sys import argv

def main():
    path = 'teste_ppa/'
    args = ['1.2.8', '1.3.8', '2.4.8', '1.4.8', '1.2.2', '1.3.2', '2.4.2', '1.4.2']
    for x in args:
        v0 = 0.0
        v1 = 0.0
        j = 0
        for i in range(0,10):
            f = open(path+x+'.'+str(i), 'r')
            
            v = f.readlines()[-1].split()

            v0 += float(v[0])
            v1 += float(v[1])
            j += 1
        print x.rjust(2), str(v0/j).rjust(3), str(v1/j).rjust(4) 
    return 0





if __name__ == "__main__":
    main()
