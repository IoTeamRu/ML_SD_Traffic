#!/bun/bash

cd $1
git clone -n --depth=1 --filter=tree:0 https://github.com/LucasAlegre/sumo-rl
cd sumo-rl
git sparse-checkout set --no-cone nets
git checkout
mv nets ..
cd ..
rm -fr ./sumo-rl
echo 'Scenarios were downloaded successfuly'
