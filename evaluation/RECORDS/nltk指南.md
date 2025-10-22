# 1) 准备目录
mkdir -p ~/nltk_data/tokenizers ~/nltk_data/corpora

# 2) 直接下载资源包（走 GitHub Raw 的稳定直链）
curl -L https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip -o /tmp/punkt.zip
unzip -o /tmp/punkt.zip -d ~/nltk_data/tokenizers

curl -L https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip -o /tmp/wordnet.zip
unzip -o /tmp/wordnet.zip -d ~/nltk_data/corpora

# （可选）把目录写进环境变量，避免 NLTK 找不到
echo 'export NLTK_DATA="$HOME/nltk_data"' >> ~/.zshrc
source ~/.zshrc

https://chatgpt.com/share/68f8f088-b410-800c-b5ba-0930173cfd84