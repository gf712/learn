#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use] extern crate rocket;
use rocket::http::RawStr;

#[get("/rank/<id>")]
fn user(id: usize) -> String  { 
	format!("It's an unsigned integer: {}", id)
}

#[get("/rank/<id>", rank = 2)]
fn user_int(id: isize) -> String { 
	format!("It's a signed integer: {}", id)
}

#[get("/rank/<id>", rank = 3)]
fn user_str(id: &RawStr) -> String {
	format!("It's a string: {}", id)
}

fn main() {
    rocket::ignite()
        .mount("/", routes![user, user_int, user_str])
        .launch();
}
