"""
Part-of-Speech(pos) tagging
"""
from functools import partial
from collections import namedtuple, defaultdict
from math import log

START_WORD = '###'

def main():
    n_w = defaultdict(lambda: 0) #Dict[str, int] Count number of word.
    n_p = defaultdict(lambda: 0) #Dict[str, int] Count number of pos.
    n_wp = defaultdict(lambda: 0) #Dict[(str, str), int] Count number of pair of word and pos. n(x,y)
    n_pp = defaultdict(lambda: 0) #Dict[(str, str), int] Count number of a pos befor a pos. n(y,y')

    dpos = START_WORD
    with open("data/entrain", 'r') as f:
        for line in f:
            word, pos = line.rstrip('\n').split('/')

            n_w[word] += 1
            n_p[pos] += 1
            n_wp[(word, pos)] += 1
            n_pp[(pos, dpos)] += 1
            dpos = pos

    print(len(n_w))
    print(n_p)

    p_wp = {} #probability of pair of word and pos. P(x|y)
    p_pp = {} #probability of a pos befor a pos. P(y|y')

    def getProb(key, p, n, noe):
        def calc(key, n, noe):
            ep = 0.01
            s=0
            for i in (n[k] for k in n if k[1]==key[1]):
            s+=i
            return (n[key]+ep)/(s+ep*noe)

        if key not in p:
            p[key] = calc(key, n, noe)
        return p[key]

    Pwp = partial(getProb, p=p_wp, n=n_wp, noe=len(n_w))
    Ppp = partial(getProb, p=p_pp, n=n_pp, noe=len(n_p))

    def viterbi(words):
        state = namedtuple('State', ['prob', 'dpos'])
        t = [{START_WORD:state(0, START_WORD)}] #List[Dict[str, State[int, str]]
        dtt = t[0]
        for x in words:
            tt = {}
            for y in n_p:
                tt[y] = max([state(log(Pwp((y, dy))*Ppp((x, y))) + dtt[dy].prob, dy) for dy in dtt])
            t.append(tt)
            dtt = tt
        print(t)
        last = max(t[-1].items(), key=lambda x:x[1])
        #print(last)
        return t

    with open("test", 'r') as f:
        word, pos = f.readline().rstrip('\n').split('/')
        words = [word]
        poss = [pos]
        for line in f:
            word, pos = line.rstrip('\n').split('/')
            if word == START_WORD:
                print(words)
                print(poss)
                p = viterbi(words)
                #print(p)
                words.clear()
                poss.clear()
            words.append(word)
            poss.append(pos)

if __name__ == '__main__':
    main()