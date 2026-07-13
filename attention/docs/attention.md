参考 dive into deeplearning 中文/英文

非意志线索是一是事物的特征，意志线索是我们主观想要干的事情，感觉输入是事物的信息
我想找钥匙是 query；钥匙扣的显眼特征是 key；钥匙这个对象携带的实际信息是 value。query 用来匹配 key，匹配上以后取出对应的 value。

注意力分数是标量，来自对每对query,key计算h(query,key)。h可以是加性注意力，点积注意力或者任何能衡量query，key之间关系的函数。h本身还要经过softmax处理。


注意力层使用注意力汇聚公式得出的输出的是 value 的加权向量。对于一个 query，将它与所有 key 的分数归一化为注意力权重，然后使用这些权重对对应的 value 做加权求和。注意力层最终输出的是 value 的加权向量。
注意力汇聚公式 f(query, (key_1,value_1), (key_2,value_2),..., (key_n,value_n)) = /sum_{i=1}^{n} h(query, key_i)*value_i