count=$1
i=0
while [ "$i" -lt "$count" ]; do
    rm -r ./R2_$i
    i=$((i+1))
done