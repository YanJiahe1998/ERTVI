count=$1
i=0
while [ "$i" -lt "$count" ]; do
    cp -r ./R2 ./R2_$i
    i=$((i+1))
done