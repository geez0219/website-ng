import { Component, OnInit } from '@angular/core';
import { NavigationStart, Router } from '@angular/router';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { MatDialog} from '@angular/material/dialog';
import { DialogComponent} from '../dialog/dialog.component'
import { BehaviorSubject } from 'rxjs';

@Component({
  selector: 'app-navbar',
  templateUrl: './navbar.component.html',
  styleUrls: ['./navbar.component.css']
})
export class NavbarComponent implements OnInit {
  isNavbarCollapsed=true;
  selected: string;
  searchContent:any
  structureHeaderDict = {
    'Content-Type': 'application/json',
    'Accept': "application/json, text/plain",
    'Access-Control-Allow-Origin': '*'
  }
  structureRequestOptions = {
    headers: new HttpHeaders(this.structureHeaderDict),
  };

  constructor(private router: Router, private http: HttpClient, public dialog: MatDialog) { 
  }

  ngOnInit() {
    this.router.events.subscribe((val) => {
      if (val instanceof NavigationStart) {
        const ns = <NavigationStart>val;
        this.selected = ns.url.substring(1).split("/")[0];
      }
    });
  }

  preRoute(newSelection: string) {
    this.isNavbarCollapsed = !this.isNavbarCollapsed;
    this.selected = newSelection.toLowerCase();
  }


  onClick(content:string){
    var httpPrefix = "https://www.googleapis.com/customsearch/v1?q=";
    var httpPostfix = "&cx=007435124061301021685%3Anx5ivx9bz4c&key=AIzaSyBqaEXf6vE07xB4PONkHzCSEb69XDCSud8";

    this.http.get(httpPrefix+content+httpPostfix, this.structureRequestOptions).subscribe(data => {
      console.log(data);

      // data.items[n].Link, Snippet, Title, 
    })
  }

  onClick2(content){
    var httpPrefix = "https://www.googleapis.com/customsearch/v1?q=";
    var httpPostfix = "&cx=007435124061301021685%3Anx5ivx9bz4c&key=AIzaSyBqaEXf6vE07xB4PONkHzCSEb69XDCSud8";

    this.http.get(httpPrefix+content+httpPostfix, this.structureRequestOptions).subscribe(data => {
      console.log(data);

      const dialogRef = this.dialog.open(DialogComponent, {
        width: '50%',
        data: data
      });
    })
  }
}
