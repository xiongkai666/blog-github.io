---
title: 一文学习Git基础命令
date: 2025-02-28 17:40:38
tags: 
- C++
- Linux
categories: 工具
---
# Git基础命令
## 安装
1. 在Linux上安装
```shell
sudo apt-get install git
```
2. 在Windows上安装：下载git程序
3. 安装完设置，在命令行输入
```shell
git config --global user.name "Your Name"
git config --global user.email "email@example.com"
```
## 创建版本库
### 创建空目录
```shell
mkdir learngit
cd learngit
pwd
```
### 将目录变成git仓库
```shell
git init
```
## 把文件添加到版本库
```shell
#1.创建并编写文件
touch 文件名
vim 文件名
#2.暂存文件，将文件放入暂存区。
git add 文件名
#3.提交更新，找到暂存区的文件，存储到Git仓库。
git commit -m "本次提交的说明"
```
## 版本回退
### 查看当前版本状态
1. 要随时掌握工作区的状态，使用`git status`命令。
2. 如果`git status`告诉你有文件被修改过，用`git diff`可以查看修改内容。
### 版本回退
1. `git log`命令显示从最近到最远的提交日志。
2. `HEAD`指向的版本就是当前版本，使用命令`git reset --hard commit_id`回退到过去的指定版本或者`git reset --hard HEAD~n`回退到过去的上`n`个版本。
3. 用git reflog查看命令历史，以便确定要回到未来的哪个版本。
### 工作区和暂存区
1. 开始创建的`learngit`就是工作区。
2. 工作区有一个隐藏目录`.git`，就是`Git`的版本库。
3. `Git`的版本库里存了很多东西，其中最重要的就是称为`stage`（或者叫`index`）的暂存区，还有`Git`为我们自动创建的第一个分支`master`，以及指向`master`的一个指针叫`HEAD`。
### 管理修改
#### 撤销修改
`git`管理的是修改，而不是文件。
1. 撤销工作区的修改，用`git checkout -- 文件名`。`git checkout`其实是用暂存区的版本替换工作区的版本。
2. 修改只是添加到了暂存区，还没有提交，先用`git reset HEAD 文件名`（回退到上个版本），把暂存区的修改回退到工作区；再使用`git checkout -- 文件名`撤销工作区的修改。
### 删除文件
1. 如果文件`commit`之后被删除，则使用`git rm`文件名或`git add 文件名`，删除暂存区的文件，再`git commit -m "此次删除说明"`。
2. 误删除文件，使用`git checkout`，把文件恢复到最新版本。
## 远程仓库
### 添加远程仓库
#### 未创建本地仓库
```shell
git init
git add `文件名`
git commit -m "first commit"
git branch -M main #默认分支名是master，将其更改为main
git remote add origin `远程仓库地址（github项目地址）` # 将本地仓库和远程仓库origin相关联
#git remote set-url origin `远程仓库地址（github项目地址）` // 重新设置地址
git push -u origin main # 将main中文件推送到origin远程仓库的main分支
```
#### 已建立本地仓库
```shell
git remote add origin `远程仓库地址（github项目地址）`
git branch -M main
git push -u origin main
```
### 从远程仓库克隆
使用`git clone git@github.com:xiongkai666/learngit.git`或`git clone https://github.com/xiongkai666/learngit.git`。
默认的`git://`使用`ssh`，但也可以使用`https`等其他协议。使用`https`除了速度慢以外，还有个最大的麻烦是每次推送都必须输入口令，但是在某些只开放`http`端口的公司内部就无法使用`ssh`协议而只能用`https`。
## 分支管理
### 创建与合并分支
1. 创建分支`git branch 分支名`。
2. 切换分支`git switch 分支名`(更优)或者`git checkout 分支名`
3. 创建并切换到分支（相当于以上两条命令）`git checkout -b 分支名`。
4. 查看所有分支`git branch`，`*`后表示当前分支。
5. `git merge 分支1`表示将分支1合并到当前分支，即保持当前分支和分支1的最新提交是完全一样的。
6. `git branch -d 分支1` 删除分支1。
>因为创建、合并和删除分支非常快，所以`Git`鼓励你使用分支完成某个任务，合并后再删掉分支，这和直接在`master`分支上工作效果是一样的，但过程更安全。
7. 合并后，用`git log`看分支历史(分支图)：`git log --graph --pretty=oneline --abbrev-commit`。
### 解决冲突
- 两个分支同时对一个文件进行修改并`commit`，如果使用`merge`进行合并，会造成冲突。解决方法：不同分支的同一文件存在冲突，必须手动解决冲突后（修改当前分支的文件）再提交，合并完再删除另一个分支。
1. Git会告诉我们哪些文件存在冲突，必须手动解决冲突后再提交。`git status`也可以告诉我们冲突的文件；
2. 查看冲突：打开对应文件，会看到类似这样的标记：
```shell 
<<<<<<< HEAD  
... 本地分支的内容 ...  
=======  
... 要合并的分支的内容 ...  
>>>>>>> [分支名]
```
3. 手动解决冲突：编辑文件，删除Git提供的冲突标记（如 <<<<<<<, =======, >>>>>>>），并决定保留哪些内容或进行其他修改；
4. 标记为已解决：一旦解决了所有冲突，使用`git add`命令来标记文件为已解决冲突的状态；
5. 提交合并：创建一个新的合并提交，包含您解决冲突后的更改：`git commit -m "合并说明"`。
6. 推送更改：最后，您可以使用`git push`命令将您的更改推送到远程仓库。
### 分支管理策略
1. `Git`默认使用`Fast forward`模式，但这种模式下，删除分支后，会丢掉分支信息。
2. 如果要强制禁用`Fast forward`模式，`Git`就会在`merge`时生成一个新的`commit`以此记录提交信息。使用`git merge --no-ff -m "merge with no-ff" dev`。
### Bug分支
1. 当前分支工作区内有文件未提交，此时需要切换到其他分支处理`bug`。`Git`提供了一个`stash`功能，可以把当前工作现场“储藏”起来，然后切换到其他分支，处理完之后再切换回来，恢复现场后继续未完成的工作。
2. 使用`git stash list`命令查看分支的“储藏区”。
3. 一是用`git stash apply`恢复，但是恢复后，`stash`内容并不删除，你需要用`git stash drop`来删除；另一种方式是用`git stash pop`，恢复的同时把`stash`内容也删了：
4. 可以多次`stash`，恢复的时候，先用`git stash list`查看，然后恢复指定的`stash`，用命令：`$ git stash apply stash@{n}`。
5. `Git`专门提供了一个`cherry-pick`命令，让我们能复制一个特定的提交到当前分支`git cherry-pick 版本号`。
6. 开发一个新功能，最好新建一个分支；如果要丢弃一个没有被合并过的分支，可以通过`git branch -D 分支名`强行删除。
### 多人协作
1. 协作伙伴从远程库`clone`时，默认情况下只能看到本地的`main`分支。
2. 要在`dev`分支上开发，就必须创建远程`origin`的`dev`分支到本地，用以下命令创建本地`dev`分支：`$ git checkout -b dev origin/dev`；就可以在`dev`上继续修改，然后，时不时地把`dev`分支`push`到远程。
3. 协作伙伴已经向origin/dev分支推送了他的提交，而碰巧你也对同样的文件作了修改，并试图推送，推送失败，因为他的最新提交和你试图推送的提交有冲突。
4. 解决办法是：先用`git pull`把最新的提交从`origin/dev`抓下来。但`git pull`失败了，原因是没有指定本地·分支与远程·分支的链接，根据提示，设置`dev`和`origin/dev`的链接：`$ git branch --set-upstream-to dev origin/dev`。
5. `git pull`成功，但是合并有冲突，需要手动解决，解决的方法和分支管理中的解决冲突完全一样。解决后，提交，再`push`。
#### 多人协作的工作模式：
1. 首先，可以试图用`git push origin <branch-name>`推送自己的修改；
2. 如果推送失败，则因为远程分支比你的本地更新，需要先用`git pull`试图合并；
3. 如果合并有冲突，则解决冲突，并在本地提交；
4. 没有冲突或者解决掉冲突后，再用`git push origin <branch-name>`推送就能成功！
5. 如果`git pull`提示`no tracking information`，则说明本地分支和远程分支的链接关系没有创建，用命令`git branch --set-upstream-to <branch-name> origin/<branch-name>`。
### rebase和merge的区别
1. `rebase`（变基）是将一个分支上的提交逐个地应用到另一个分支上，使得提交历史变得更加线性。当执行`rebase`时，`git`会将目标分支与源分支的共同祖先以来的所有提交挪到目标分支的最新位置。这个过程可以看作是将源分支上的每个提交复制到目标分支上。简而言之，`rebase`可以将提交按照时间顺序线性排列。
2. `merge`（合并）是将两个分支上的代码提交历史合并为一个新的提交。在执行`merge`时，`git`会创建一个新的合并提交，将两个分支的提交历史连接在一起。这样，两个分支的修改都会包含在一个合并提交中。合并后的历史会保留每个分支的提交记录。
## 标签
- 发布一个版本时，通常先在版本库中打一个标签（tag，相当于版本号），就唯一确定了打标签时刻的版本。将来无论什么时候，取某个标签的版本，就是把那个打标签的时刻的历史版本取出来。所以，标签也是版本库的一个快照。
### 创建标签
1. `git tag <tag name>`就可以打一个新标签，默认标签是打在最新提交的`commit`上的；
2. 给历史`commit`打标签的方法是：找到历史提交的`commit id`，然后敲入命令`git tag <tag name> <commit id>`。
3. 还可以创建带有说明的标签，用-a指定标签名，-m指定说明文字：如`git tag -a v0.1 -m "version 0.1 released" 1094adb`。
### 操作标签
1. 因为创建的标签都只存储在本地，不会自动推送到远程。所以，打错的标签可以在本地安全删除。使用`git tag -d <tag name>`删除。
2. 如果要推送某个标签到远程，使用命令`git push origin <tag name>`。
3. 一次性推送全部尚未推送到远程的本地标签：`git push origin --tags`。
4. 如果标签已经推送到远程，要删除远程标签就麻烦一点，先从本地删除：`git tag -d <tag name>`；然后，从远程删除。删除命令也是`push`，格式如下：`git push origin :refs/tags/<tag name>`。
5. 要看是否真的从远程库删除了标签，可以登陆`GitHub`查看。
### 参加github开源项目
1. fork他人仓库到自己账号；
2. 在自己的帐号下将该仓库clone到本地；
3. 在本地仓库修改并推送到自己的远程仓库；
4. 进入`github`帐号找到修改的地方，可以推送`pull request`给官方仓库来贡献代码。
### 下载github项目的历史版本
1. `git clone` 项目网址；
2. 点击`commit`；
3. 找到自己想下载的版本，复制`SHA`；
4. `cd`到刚刚下载的代码目录；
5. `git checkout SHA`。
## 自定义git
### 忽略特殊文件
1. 忽略某些文件时，需要编写`.gitignore`；
2. `.gitignore`文件本身要放到版本库里，并且可以对`.gitignore`做版本管理；
3. 文件`.gitignore`的格式规范：
   - `#`为注释； 
   - 可以使用`shell`所使用的正则表达式来进行模式匹配  ； 
   - 匹配模式最后跟`/`说明要忽略的是目录；
   - 使用`！`取反(表示不忽略)。
### 配置别名
1. 有些`git`命令不好记，可以使用别名来替代。语法：`git config --global alias.别名 原命令`。
2. 很多人都用`ci`表示`commit`：`git config --global alias.ci commit`；以后提交就可以简写成：`git ci -m "bala..."`。
### 常见错误
1. 错误：`Failed to connect to github.com port 443 : Timed out`
解决方法：修改网络设置
```shell
git config --global http.proxy http://127.0.0.1:7890 
git config --global https.proxy http://127.0.0.1:7890
```
1. 错误：`fatal: unable to access '**网址': Recv failure: Connection was reset`
解决方法：取消代理
```shell
git config --global --unset http.proxy
git config --global --unset https.proxy
```