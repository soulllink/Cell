
#[test]
fn test_parse_fizzbuzz_while() {
    let input = "
      i = 1
      while i < 16 do
        put(i)
        i = i + 1
      end
    ";
    let res = parse_cell_content(input);
    assert!(res.is_ok(), "Failed to parse while loop: {:?}", res.err());
}
