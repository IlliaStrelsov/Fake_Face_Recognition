drop table if exists users;
drop table if exists attendance;

create table IF NOT EXISTS users
(
    id            serial
            primary key,
    name varchar not null,
    password varchar not null
);

create table IF NOT EXISTS attendance
(
    id            serial
            primary key,
    user_name varchar not null,
    date timestamp not null default CURRENT_TIMESTAMP
);

alter table users
    owner to postgres;

alter table attendance
    owner to postgres;