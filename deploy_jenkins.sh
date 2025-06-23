#!/bin/bash

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use node

cd /work/cursor

git pull;

SRC=/work/cursor/node/hello
DEST=$HOME/deploy_jenkins

rm -rf $DEST
mkdir -p $DEST
cp -r $SRC $DEST

cd $DEST/hello

npm install

npm install -g pm2

pm2 restart all
