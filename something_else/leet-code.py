class Solution:

    s = set()
    def f(self, sub_exp):
        res = []
        j = sub_exp.find('}')
        if j == -1:
            self.s.add(sub_exp)
            return 
        i = sub_exp.find('{')
        prefix, middle, postfix = sub_exp[:i], sub_exp[i + 1: j],sub_exp[j + 1:]
        if prefix != '':
            self.s.add(prefix)
        if postfix != '':
            self.s.add(postfix)
        self.f(middle)

    def braceExpansionII(self, expression: str):
        self.f(expression)

if __name__ == "__main__":
    Solution().braceExpansionII("{a,b}{c,{d,e}}")
