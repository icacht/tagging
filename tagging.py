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

    p_wp = {} #probability of pair of word and pos. P(x|y)
    p_pp = {} #probability of a pos befor a pos. P(y|y')

    def getProb(key, p, n, nn, noe):
        def calc(key, n, nn, noe):
            ep = 0.0000001
            return (n[key]+ep)/(nn[key[1]]+ep*noe)

        if key not in p:
            p[key] = calc(key, n, nn, noe)
        return p[key]

    Pwp = partial(getProb, p=p_wp, n=n_wp, nn=n_p, noe=len(n_w))
    Ppp = partial(getProb, p=p_pp, n=n_pp, nn=n_p, noe=len(n_p))

    def viterbi(words):
        state = namedtuple('State', ['prob', 'dpos'])
        t = [{START_WORD:state(0, START_WORD)}] #List[Dict[str, State[int, str]]
        dtt = t[0]
        for x in words:
            tt = {}
            for y in n_p:
                tt[y] = max([state(log(Pwp((x, y))*Ppp((y, dy))) + dtt[dy].prob, dy) for dy in dtt])
            t.append(tt)
            dtt = tt

        mt = max(t[-1].items(), key=lambda x:x[1])
        dpos = mt[1].dpos
        l = [mt[0]]
        for i in reversed(t[:-1]):
            l.append(dpos)
            dpos = i[dpos].dpos
        l.reverse()
        return l

    with open("data/entest", 'r') as f:
        word, pos = f.readline().rstrip('\n').split('/')
        words = [word]
        poss = [pos]
        cdiff = 0
        cwords = 0
        for line in f:
            word, pos = line.rstrip('\n').split('/')
            if word == START_WORD:
                p = viterbi(words[1:])
                diff = [i for i,(x,y) in enumerate(zip(p, poss)) if x != y]
                cdiff += len(diff)
                cwords += len(words)
                print(words)
                print(poss)
                print(p)
                print(diff)
                print(len(diff), '/', len(words))
                words.clear()
                poss.clear()
            words.append(word)
            poss.append(pos)
        print('All:', cdiff, '/', cwords, 'correct:', (1-(cdiff/cwords))*100, '%')

if __name__ == '__main__':
    main()