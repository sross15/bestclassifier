context("argument requirements")

test_that("data is a data frame with at least two columns and rows", {
  expect_error(class(data) != data.frame)
})

test_that("repeats is either NA or a whole number greater than 0", {
  expect_error(is.character(repeats))
})

test_that("form must be a valid formula", {
  expect_error(class(form) != "formula")
})

test_that("model is a character vector assigned one or more of the models", {
  expect_error(class(model) != "character")
} )

test_that("set_seed is a numberic object", {
  expect_error(!is.numeric(set_seed))
})

test_that("desired_metric is a character object", {
  expect_error(class(desired_metric) != "character")
})