
本项目通过 hexo + github + netlify + algolia 完成自动部署的博客网站搭建 

## 生成本地博客
```shell
# 切换到根目录下
1. hexo clean
2. hexo algolia #生成搜索功能
3. hexo g
4. hexo s
```

## 博客部署到服务器
### 更新主题子模块
```shell
# 切换到根目录下
git submodule update --remote themes/butterfly
```
### 2. 生成博客网页
```shell
# 和github提交操作相同
1. git add .
2. git commit -m ""
3. git push
```
### 3. 访问网址``https://kkblog.top/``
### 4. 域名续费
网址：https://wanwang.aliyun.com/buy/commonbuy?scenario=renew&domainCartTradeParams=dr_12818093_47_1751796851436&orderId=RO20250706181423000001