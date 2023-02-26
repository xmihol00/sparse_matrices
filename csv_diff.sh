column_count=$(head -n 1 $1 | tr ',' '\n' | wc -l)
chars_per_cell=$(head -n 1 $1 | tr ',' '\n' | head -n 1 | wc -c)
difference=$(diff <(cat $1 | tr ',' '\n') <(cat $2 | tr ',' '\n'))
#difference=$(diff <(head -n 1000 $1 | tr ',' '\n') <(head -n 1000 $2 | tr ',' '\n'))
rows=($(echo $difference | tr ' ' '\n' | grep -E -i '[0-9]+c[0-9]+'))
values=($(echo $difference | tr ' ' '\n' | grep -E -i '[0-9]+\.[0-9]+'))

indices=""
for i in ${rows[@]}
do
  row_index=$(( ${i%%c*} / $column_count ))
  indices="$indices $(( $row_index + 1))x$(( (${i%%c*} - $row_index * $column_count) * $chars_per_cell - $chars_per_cell + 1 ))-$(( (${i%%c*} - $row_index * $column_count) * $chars_per_cell - 1 ))"
done
indices=($indices)

for i in ${!indices[*]}
do
    echo -e "${indices[$i]}:\t${values[$(( $i * 2 ))]} ${values[$(( $i * 2 + 1 ))]}"
done
